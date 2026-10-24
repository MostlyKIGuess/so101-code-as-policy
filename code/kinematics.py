"""Forward and inverse kinematics of the SO-101 in bus units, through so101/urdf_map.json.

Positions are in centimetres in the model frame: the base plate's bottom is z = 0,
which is the table, -y points forward out of the arm (towards the camera), and x
is lateral. The model is so101/so_101.xml, which has the URDF's joints. The tip
is the farthest-forward vertex of the fixed jaw, fixed in that jaw's frame.
"""

import json

import mujoco
import numpy as np

from cap_so101.arm import JOINT_NAMES, REPO, Arm

MODEL = REPO / "so101" / "so_101.xml"
# Above the pen and the mouse (about 4 cm), below the cat toy (about 10 cm), so no
# path is planned over the cushion.
TRAVEL_CM = 8.0
# Lowest crossing allowed where 8 cm is out of reach: 2 cm over the mouse, the tallest
# object a crossing passes over.
TRAVEL_FLOOR_CM = 6.0
# Read on 24 Oct with the arm folded (runs/session2-start.jpg). Roll and gripper are kept as read.
REST = {"shoulder_pan": -26.0, "shoulder_lift": -105.8, "elbow_flex": 103.0, "wrist_flex": 72.4}
URDF_MAP = REPO / "so101" / "urdf_map.json"
# The base and the pan bracket sit on the table and are left out of the lowest point.
MOVING_BODIES = ["Upper_Arm", "Lower_Arm", "Wrist_Pitch_Roll", "Fixed_Jaw", "Moving_Jaw"]
# Joints the inverse kinematics solves for. Wrist roll and gripper stay as given.
IK_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]


class Kinematics:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        urdf_map = json.loads(URDF_MAP.read_text())
        self.entries = [urdf_map[name] for name in JOINT_NAMES]
        self.qpos_index = [self.model.jnt_qposadr[self.model.joint(entry["urdf"]).id] for entry in self.entries]

        # Mesh vertices of the moving bodies, each in its geom's frame.
        self.geoms = []
        for geom in range(self.model.ngeom):
            body_name = self.model.body(self.model.geom_bodyid[geom]).name
            if body_name in MOVING_BODIES and self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = self.model.geom_dataid[geom]
                start = self.model.mesh_vertadr[mesh]
                self.geoms.append((geom, self.model.mesh_vert[start:start + self.model.mesh_vertnum[mesh]].copy()))

        # The tip: at the model's zero pose the jaw points along -y, so the tip is the
        # fixed jaw vertex with the smallest y, stored in the Fixed_Jaw body frame.
        mujoco.mj_forward(self.model, self.data)
        self.jaw_body = self.model.body("Fixed_Jaw").id
        self.roll_joint = self.model.joint("Wrist_Roll").id
        # Where the vertical pan axis meets the table plane, in cm. The arm can only point
        # within a vertical plane through it, and it is not at the origin.
        self.pan_axis_cm = self.data.xanchor[self.model.joint("Rotation").id][:2] * 100
        jaw_world = np.vstack([self._world_vertices(geom, vertices) for geom, vertices in self.geoms
                               if self.model.geom_bodyid[geom] == self.jaw_body])
        tip_world = jaw_world[np.argmin(jaw_world[:, 1])]
        rotation = self.data.xmat[self.jaw_body].reshape(3, 3)
        self.tip_local = rotation.T @ (tip_world - self.data.xpos[self.jaw_body])
        # The moving jaw's tip, found the same way in its own frame.
        self.moving_body = self.model.body("Moving_Jaw").id
        moving_world = np.vstack([self._world_vertices(geom, vertices) for geom, vertices in self.geoms
                                  if self.model.geom_bodyid[geom] == self.moving_body])
        moving_tip_world = moving_world[np.argmin(moving_world[:, 1])]
        rotation = self.data.xmat[self.moving_body].reshape(3, 3)
        self.moving_tip_local = rotation.T @ (moving_tip_world - self.data.xpos[self.moving_body])

    def _world_vertices(self, geom: int, vertices: np.ndarray) -> np.ndarray:
        return vertices @ self.data.geom_xmat[geom].reshape(3, 3).T + self.data.geom_xpos[geom]

    def _set(self, joints: np.ndarray) -> None:
        for index, entry, value in zip(self.qpos_index, self.entries, joints):
            self.data.qpos[index] = np.deg2rad(entry["sign"] * entry["scale"] * value + entry["offset_deg"])
        mujoco.mj_kinematics(self.model, self.data)

    def tip(self, joints: np.ndarray) -> np.ndarray:
        """Tip position in cm for bus joints."""
        self._set(joints)
        rotation = self.data.xmat[self.jaw_body].reshape(3, 3)
        return (self.data.xpos[self.jaw_body] + rotation @ self.tip_local) * 100

    def moving_tip(self, joints: np.ndarray) -> np.ndarray:
        """Moving jaw tip position in cm for bus joints."""
        self._set(joints)
        rotation = self.data.xmat[self.moving_body].reshape(3, 3)
        return (self.data.xpos[self.moving_body] + rotation @ self.moving_tip_local) * 100

    def opening_direction(self, joints: np.ndarray) -> np.ndarray:
        """Horizontal unit vector from the fixed jaw tip towards the moving jaw tip."""
        across = self.moving_tip(joints)[:2] - self.tip(joints)[:2]
        return across / np.linalg.norm(across)

    def jaw_direction(self, joints: np.ndarray) -> np.ndarray:
        """Unit vector along the wrist roll axis, towards the tip.

        Not the line from the wrist to the tip: the tip sits 0.9 cm off the roll
        axis, so that line tilts sideways as roll turns and wrist flex cannot
        straighten it. The model's roll axis points away from the tip, hence the sign.
        """
        self._set(joints)
        return -self.data.xaxis[self.roll_joint].copy()

    def lowest(self, joints: np.ndarray) -> float:
        """Lowest point of the moving parts in cm. The table is 0."""
        self._set(joints)
        return float(min(self._world_vertices(geom, vertices)[:, 2].min() for geom, vertices in self.geoms) * 100)

    def path_lowest(self, start: np.ndarray, target: np.ndarray, samples: int = 25) -> float:
        """Lowest point along Arm.move's straight line in joint space."""
        return min(self.lowest(start + (target - start) * step / samples) for step in range(samples + 1))

    def solve_tip_down(self, target_cm: np.ndarray, seed: np.ndarray, limits: dict[str, list[float]]) -> np.ndarray:
        """Bus joints that put the tip at target_cm with the jaw pointing straight down."""
        return self.solve_pose(target_cm, np.array([0.0, 0.0, -1.0]), seed, limits)

    def solve_pose(self, target_cm: np.ndarray, direction: np.ndarray, seed: np.ndarray,
                   limits: dict[str, list[float]]) -> np.ndarray:
        """Bus joints that put the tip at target_cm with the jaw along `direction`.

        Starts from the seed, and from the rest pose if that fails: near the edge of
        reach a seed with the wrist turned ended in a local minimum (dry run, 24 Oct)
        where the rest pose solved. Raises if neither converges.
        """
        try:
            return self._solve_pose(target_cm, direction, seed, limits)
        except ValueError:
            rest = np.asarray(seed, dtype=np.float64).copy()
            for name, value in REST.items():
                rest[JOINT_NAMES.index(name)] = value
            return self._solve_pose(target_cm, direction, rest, limits)

    def _solve_pose(self, target_cm: np.ndarray, direction: np.ndarray, seed: np.ndarray,
                    limits: dict[str, list[float]], iterations: int = 200) -> np.ndarray:
        """Gauss-Newton on pan, lift, elbow and wrist flex with a numeric Jacobian.

        Raises if it does not converge to 1 mm and 1 degree inside the joint limits.
        """
        indices = [JOINT_NAMES.index(name) for name in IK_JOINTS]
        lower = np.array([limits[name][0] for name in IK_JOINTS])
        upper = np.array([limits[name][1] for name in IK_JOINTS])
        joints = np.asarray(seed, dtype=np.float64).copy()
        direction = np.asarray(direction, dtype=np.float64) / np.linalg.norm(direction)

        def residual(values: np.ndarray) -> np.ndarray:
            trial = joints.copy()
            trial[indices] = values
            position = self.tip(trial) - target_cm
            # Direction error in cm per unit, so it weighs like position.
            return np.concatenate([position, (self.jaw_direction(trial) - direction) * 10])

        values = joints[indices]
        for _ in range(iterations):
            error = residual(values)
            if np.abs(error[:3]).max() < 0.1 and np.abs(error[3:]).max() < 10 * np.sin(np.deg2rad(1)):
                break
            jacobian = np.zeros((6, 4))
            for column in range(4):
                step = np.zeros(4)
                step[column] = 0.1
                jacobian[:, column] = (residual(values + step) - error) / 0.1
            update = np.linalg.lstsq(jacobian.T @ jacobian + 1e-3 * np.eye(4), -jacobian.T @ error, rcond=None)[0]
            values = np.clip(values + np.clip(update, -10, 10), lower, upper)
        else:
            raise ValueError(f"no pose with the tip at {np.round(target_cm, 1)} cm and the jaw along "
                             f"{np.round(direction, 2)}, error {np.round(error, 2)}")
        joints[indices] = values
        return joints

    def tilted_down(self, point_cm: np.ndarray, tilt_deg: float) -> np.ndarray:
        """Straight down, leaned tilt_deg outward, away from the pan axis, at a table point.

        Leaning the jaw outward keeps the wrist nearer the base, which reaches farther:
        on 24 Oct a point 23.9 cm out on the arm's left was out of reach straight down
        above 1.5 cm and in reach from 1.5 to 8 cm with 10 degrees.
        """
        outward = np.asarray(point_cm, dtype=np.float64)[:2] - self.pan_axis_cm
        outward /= np.linalg.norm(outward)
        tilt = np.deg2rad(tilt_deg)
        return np.r_[np.sin(tilt) * outward, -np.cos(tilt)]

    def solve_grasp(self, centre_cm: np.ndarray, axis: np.ndarray, height_cm: float, side_cm: float,
                    seed: np.ndarray, limits: dict[str, list[float]], rounds: int = 10,
                    tilt_deg: float = 0.0) -> np.ndarray:
        """Tip-down joints with the jaws opening across `axis` around an object at centre_cm.

        The fixed jaw tip goes side_cm from the centre, away from the moving jaw, so
        closing the moving jaw presses the object against the fixed jaw. Each round
        turns wrist roll by the angle between the opening and the nearer of the two
        directions across the axis that roll can reach. A step on the dot product
        stalls when the opening starts along the axis, where its slope is zero.
        The seed's gripper value sets how far the jaws are open. tilt_deg leans the jaw
        outward (tilted_down) instead of straight down.
        """
        roll = JOINT_NAMES.index("wrist_roll")
        low, high = limits["wrist_roll"]
        axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
        across_axis = np.array([-axis[1], axis[0]])
        joints = np.asarray(seed, dtype=np.float64).copy()

        def heading(vector: np.ndarray) -> float:
            return float(np.degrees(np.arctan2(vector[1], vector[0])))

        for _ in range(rounds):
            opening = self.opening_direction(joints)
            target = np.r_[np.asarray(centre_cm) - side_cm * opening, height_cm]
            joints = self.solve_pose(target, self.tilted_down(target, tilt_deg), joints, limits)
            opening = self.opening_direction(joints)
            if abs(float(opening @ axis)) < np.sin(np.deg2rad(2)) and np.linalg.norm(self.tip(joints) - target) < 0.15:
                return joints
            nudged = joints.copy()
            nudged[roll] += 5.0
            turn_per_degree = ((heading(self.opening_direction(nudged)) - heading(opening) + 180) % 360 - 180) / 5.0
            if abs(turn_per_degree) < 0.1:
                raise ValueError("wrist roll does not turn the jaw opening at this pose")
            candidates = []
            for wanted in (across_axis, -across_axis):
                angle = (heading(wanted) - heading(opening) + 180) % 360 - 180
                candidate = joints[roll] + angle / turn_per_degree
                if low <= candidate <= high:
                    candidates.append((abs(angle), candidate))
            if not candidates:
                raise ValueError(f"no wrist roll inside {low} to {high} turns the jaws across axis {np.round(axis, 2)}")
            joints[roll] = min(candidates)[1]
        raise ValueError(f"no grasp around {np.round(centre_cm, 1)} cm across axis {np.round(axis, 2)}")


def checked_move(arm: Arm, kinematics: Kinematics, target: np.ndarray) -> np.ndarray:
    """Move only if no part of the arm dips below both ends of the path."""
    start = arm.read()
    floor = min(kinematics.lowest(start), kinematics.lowest(target)) - 0.5
    lowest = kinematics.path_lowest(start, target)
    if lowest < floor:
        raise RuntimeError(f"path dips to {lowest:.1f} cm, below both ends ({floor + 0.5:.1f} cm)")
    arm.set_torque(True)
    after = arm.move(target)
    print("target", np.round(target, 1).tolist())
    print("after ", np.round(after, 1).tolist(), "error", np.round(after - target, 1).tolist())
    print("tip from read-back", np.round(kinematics.tip(after), 1).tolist(), "cm")
    return after


def travel(arm: Arm, kinematics: Kinematics, final: np.ndarray) -> list[np.ndarray]:
    """Rise to TRAVEL_CM above the start, cross at that height, then go down to final.

    On 24 Oct a direct move out of rest swept the gripper sideways about 2 cm above
    the table and dragged the pen (runs/table-points/215901-hover-20--10.jpg).
    Returns the joints read back after each leg.
    """
    start = arm.read()
    start_tip, final_tip = kinematics.tip(start), kinematics.tip(final)
    up = highest_tip_down(kinematics, start_tip[:2], start, arm.limits)
    try:
        across = highest_tip_down(kinematics, final_tip[:2], up, arm.limits)
    except ValueError:
        if final_tip[2] < TRAVEL_FLOOR_CM:
            raise
        # No jaw-down pose over the destination, but it is at crossing height itself,
        # e.g. a tilted approach far out on the arm's left.
        return [checked_move(arm, kinematics, leg) for leg in (up, final)]
    return [checked_move(arm, kinematics, leg) for leg in (up, across, final)]


def checked_follow(arm: Arm, kinematics: Kinematics, waypoints: list[np.ndarray]) -> np.ndarray:
    """Arm.follow, refused before any motion if a segment dips below both of its ends."""
    start = arm.read()
    for target in waypoints:
        floor = min(kinematics.lowest(start), kinematics.lowest(target)) - 0.5
        lowest = kinematics.path_lowest(start, target)
        if lowest < floor:
            raise RuntimeError(f"segment dips to {lowest:.1f} cm, below both ends ({floor + 0.5:.1f} cm)")
        start = target
    arm.set_torque(True)
    after = arm.follow(waypoints)
    print("follow", len(waypoints), "waypoints, after", np.round(after, 1).tolist(),
          "error", np.round(after - waypoints[-1], 1).tolist())
    return after


def highest_tip_down(kinematics: Kinematics, point_cm: np.ndarray, seed: np.ndarray,
                     limits: dict[str, list[float]]) -> np.ndarray:
    """Jaw-down pose over point_cm at TRAVEL_CM, or lower in 0.5 cm steps down to TRAVEL_FLOOR_CM.

    Near the edge of reach 8 cm up has no pose (dry run, 24 Oct: 23.8 cm out on the
    arm's left), while lower ones do.
    """
    for height in np.arange(TRAVEL_CM, TRAVEL_FLOOR_CM - 0.01, -0.5):
        try:
            return kinematics.solve_tip_down(np.r_[point_cm, height], seed, limits)
        except ValueError:
            continue
    raise ValueError(f"no jaw-down pose between {TRAVEL_FLOOR_CM} and {TRAVEL_CM} cm over {np.round(point_cm, 1)}")
