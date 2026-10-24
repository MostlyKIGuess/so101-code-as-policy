"""The desk as objects in table centimetres, read from a camera frame, and its MuJoCo recreation.

Positions are in the model frame of cap_so101/kinematics.py. Flat objects (pen,
cushion, remote) map through the homography pixel by pixel. The mouse stands
about 4 cm tall, so only its lowest pixel in each column is taken: that outline
touches the table and maps truly, the rest of the mask does not.

Heights and the cat toy's size were not measured. They are read by eye off
setup/side.png and used only to draw the recreation and to keep paths above them.
"""

import json
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from cap_so101 import detect
from cap_so101.arm import REPO
from cap_so101.kinematics import MODEL, Kinematics

HOMOGRAPHY = REPO / "so101" / "table_homography.json"
PEN_RADIUS_CM = 0.55
# By eye off setup/side.png, not measured.
HEIGHT_CM = {"mouse": 4.0, "cushion": 2.0, "remote": 2.0, "cat": 10.0}
RGBA = {
    "mouse": [0.26, 0.52, 0.96, 1], "pen": [0.20, 0.66, 0.33, 1], "remote": [0.92, 0.26, 0.21, 1],
    "cushion": [0.98, 0.74, 0.02, 1], "cat": [0.95, 0.95, 0.95, 1], "table": [0.82, 0.82, 0.80, 1],
}


def load_homography() -> np.ndarray:
    return np.array(json.loads(HOMOGRAPHY.read_text())["image_to_table_cm"], dtype=np.float64)


def to_table(homography: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    points = np.column_stack([pixels, np.ones(len(pixels))]) @ homography.T
    return points[:, :2] / points[:, 2:]


def to_pixel(homography: np.ndarray, points_cm: np.ndarray) -> np.ndarray:
    points = np.column_stack([points_cm, np.ones(len(points_cm))]) @ np.linalg.inv(homography).T
    return points[:, :2] / points[:, 2:]


@dataclass
class Footprint:
    """A box on the table: centre and half sizes in cm along its own axes, turned by angle_deg."""
    centre: np.ndarray
    half: np.ndarray
    angle_deg: float

    def nearest(self, point_cm: np.ndarray) -> np.ndarray:
        """The point of the box closest to point_cm."""
        turn = np.deg2rad(self.angle_deg)
        axes = np.array([[np.cos(turn), np.sin(turn)], [-np.sin(turn), np.cos(turn)]])
        local = np.clip(axes @ (np.asarray(point_cm) - self.centre), -self.half, self.half)
        return self.centre + axes.T @ local


def footprint(points_cm: np.ndarray) -> Footprint:
    (cx, cy), (width, height), angle = cv2.minAreaRect(points_cm.astype(np.float32))
    return Footprint(np.array([cx, cy]), np.array([width, height]) / 2, float(angle))


def mouse_outline(mask: np.ndarray) -> np.ndarray:
    """Lowest mask pixel of every column, where the mouse meets the table."""
    ys, xs = np.nonzero(mask)
    columns = np.unique(xs)
    return np.array([[column, ys[xs == column].max()] for column in columns], dtype=np.float64)


@dataclass
class Desk:
    pen: tuple[np.ndarray, np.ndarray] | None = None  # the two ends in cm
    mouse_front: np.ndarray | None = None              # the outline that meets the table, in cm
    cushion: Footprint | None = None
    remote: Footprint | None = None
    pen_held: bool = False

    @classmethod
    def from_detections(cls, found: dict[str, detect.Detection], homography: np.ndarray) -> "Desk":
        desk = cls()
        if "pen" in found:
            ends = detect.ends(found["pen"])
            desk.pen = tuple(to_table(homography, np.array(ends)))
        if "mouse" in found:
            desk.mouse_front = to_table(homography, mouse_outline(found["mouse"].mask))
        for name in ("cushion", "remote"):
            if name in found:
                ys, xs = np.nonzero(found[name].mask)
                pixels = np.column_stack([xs, ys])[:: max(1, len(xs) // 4000)]
                setattr(desk, name, footprint(to_table(homography, pixels.astype(np.float64))))
        return desk

    def mouse_edges(self) -> tuple[float, float, np.ndarray]:
        """Left and right x of the mouse's front outline, and its middle point."""
        if self.mouse_front is None:
            raise RuntimeError("no mouse on the desk")
        return float(self.mouse_front[:, 0].min()), float(self.mouse_front[:, 0].max()), self.mouse_front.mean(axis=0)


class Recreation:
    """The arm model with the desk objects as mocap bodies, rendered offscreen."""

    def __init__(self, kinematics: Kinematics, width: int = 640, height: int = 480) -> None:
        spec = mujoco.MjSpec.from_file(str(MODEL))
        spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
        spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.6, 0.6, 0.01], rgba=RGBA["table"])
        self.names = ["pen", "mouse", "cushion", "cat", "remote"]
        shapes = {"pen": mujoco.mjtGeom.mjGEOM_CAPSULE, "mouse": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                  "cushion": mujoco.mjtGeom.mjGEOM_BOX, "cat": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                  "remote": mujoco.mjtGeom.mjGEOM_BOX}
        for name in self.names:
            body = spec.worldbody.add_body(name=name, mocap=True, pos=[0, 0, -1])
            body.add_geom(name=name, type=shapes[name], size=[0.01, 0.01, 0.01], rgba=RGBA[name],
                          contype=0, conaffinity=0)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.kinematics = kinematics
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.camera = mujoco.MjvCamera()
        # Looking from the camera's side of the desk, a little to the arm's left.
        self.camera.lookat[:] = [0.0, -0.2, 0.03]
        self.camera.distance = 0.75
        self.camera.elevation = -40
        self.camera.azimuth = 70

    def _place(self, name: str, centre_cm: np.ndarray, half_cm: np.ndarray, angle_deg: float) -> None:
        body = self.model.body(name).mocapid[0]
        geom = self.model.geom(name).id
        self.data.mocap_pos[body] = np.asarray(centre_cm) / 100
        turn = np.deg2rad(angle_deg) / 2
        self.data.mocap_quat[body] = [np.cos(turn), 0, 0, np.sin(turn)]
        self.model.geom_size[geom] = np.asarray(half_cm) / 100

    def _hide(self, name: str) -> None:
        self.data.mocap_pos[self.model.body(name).mocapid[0]] = [0, 0, -1]

    def render(self, joints: np.ndarray, desk: Desk) -> np.ndarray:
        """RGB image of the arm at bus joints with the desk objects."""
        for entry, value in zip(self.kinematics.entries, joints):
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(entry["urdf"]).id]] = np.deg2rad(
                entry["sign"] * entry["scale"] * value + entry["offset_deg"])
        mujoco.mj_kinematics(self.model, self.data)
        if desk.pen is not None:
            end_a, end_b = desk.pen
            if desk.pen_held:
                # Drawn between the jaw tips, lying across the opening.
                grip = (self.kinematics.tip(joints) + self.kinematics.moving_tip(joints)) / 2
                across = self.kinematics.opening_direction(joints)
                along = np.array([-across[1], across[0]])
                length = float(np.linalg.norm(end_b - end_a))
                end_a, end_b = grip[:2] - along * length / 2, grip[:2] + along * length / 2
                height = grip[2]
            else:
                height = PEN_RADIUS_CM
            middle = (end_a + end_b) / 2
            heading = np.degrees(np.arctan2(*(end_b - end_a)[::-1]))
            # A capsule lies along its own z, so turn it onto x first.
            body = self.model.body("pen").mocapid[0]
            self.data.mocap_pos[body] = np.r_[middle, height] / 100
            yaw, pitch = np.deg2rad(heading) / 2, np.deg2rad(90) / 2
            yaw_quat = np.array([np.cos(yaw), 0, 0, np.sin(yaw)])
            pitch_quat = np.array([np.cos(pitch), 0, np.sin(pitch), 0])
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, yaw_quat, pitch_quat)
            self.data.mocap_quat[body] = quat
            self.model.geom_size[self.model.geom("pen").id] = [PEN_RADIUS_CM / 100, float(np.linalg.norm(end_b - end_a)) / 200, 0]
        else:
            self._hide("pen")
        if desk.mouse_front is not None:
            left, right, middle = desk.mouse_edges()
            # The outline is the side facing the camera, the most negative y. The body runs
            # back from there. Ten cm long is a typical mouse, not measured.
            half = np.array([(right - left) / 2, 5.0, HEIGHT_CM["mouse"] / 2])
            centre = np.array([(left + right) / 2, desk.mouse_front[:, 1].min() + half[1], half[2]])
            self._place("mouse", centre, half, 0.0)
        else:
            self._hide("mouse")
        if desk.cushion is not None:
            cushion = desk.cushion
            self._place("cushion", np.r_[cushion.centre, HEIGHT_CM["cushion"] / 2],
                        np.r_[cushion.half, HEIGHT_CM["cushion"] / 2], cushion.angle_deg)
            self._place("cat", np.r_[cushion.centre, HEIGHT_CM["cushion"] + HEIGHT_CM["cat"] / 2],
                        np.r_[cushion.half * 0.6, HEIGHT_CM["cat"] / 2], cushion.angle_deg)
        else:
            self._hide("cushion")
            self._hide("cat")
        if desk.remote is not None:
            remote = desk.remote
            self._place("remote", np.r_[remote.centre, HEIGHT_CM["remote"] / 2],
                        np.r_[remote.half, HEIGHT_CM["remote"] / 2], remote.angle_deg)
        else:
            self._hide("remote")
        mujoco.mj_kinematics(self.model, self.data)
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()
