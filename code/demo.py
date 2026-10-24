"""The live robot server for the "Demo video" section of PLAN.md, logged to Rerun.

`serve` holds the arm, the camera, the MuJoCo model and the Rerun stream open. The
agent sends one step of Python at a time with scripts/act.py, usually a call to a
skill in cap_so101/skills.py, reads what it returns and the new frame, and decides
the next step. Every step's code is logged before it runs. The recording holds, on
one timeline: the goal and the subtask list (ticked by the agent with demo.mark_done
when the camera confirms), each step's code, the camera frame with the objects, the
planned path and the predicted end box, the arm on its URDF, a MuJoCo recreation of
the desk, the perception method, and predicted against measured push displacement.

    python scripts/demo.py --live serve --screenshots       # real arm
    python scripts/demo.py --live serve --dry-run           # simulated arm, real camera
    python scripts/act.py "look()"

--dry-run reads the arm and the camera but sends no motor command: a simulated arm
takes every move, objects move only as a skill expects, and a push moves the mouse
DRY_PUSH_GAIN times the travel after contact. --live also streams to a viewer started
with `rerun`. A run lives in runs/demo/<stamp>-live/.
"""

import argparse
import ast
import contextlib
import functools
import io
import json
import socket
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.urdf import UrdfTree

from cap_so101 import detect, sam, skills
from cap_so101.arm import JOINT_NAMES, REPO, Arm
from cap_so101.camera import Camera
from cap_so101.kinematics import Kinematics, checked_follow, checked_move, travel
from cap_so101.rerun_urdf import log_joint_angles
from cap_so101.scene import Desk, Recreation, load_homography, to_pixel, to_table

APP_ID = "cap-so101-demo"
# scripts/act.py sends steps here.
SOCKET = Path("/tmp/cap-so101-robot.sock")
URDF = REPO / "so101" / "so101.urdf"
URDF_MAP = REPO / "so101" / "urdf_map.json"
# In a dry run a push moves the simulated mouse this times the gripper's travel after
# contact. Below the starting gain of 1, so the first push falls short and the refit
# and a second push are rehearsed.
DRY_PUSH_GAIN = 0.7

# One palette for every view, RGB.
COLOURS = {
    "mouse": (66, 133, 244),
    "pen": (52, 168, 83),
    "remote": (234, 67, 53),
    "cushion": (251, 188, 5),
    "arm": (120, 120, 120),
    "predicted": (171, 71, 188),
    "path": (171, 71, 188),
    "measured": (66, 133, 244),
}

METHOD_SAM = """\
### Perception with SAM (on in this run)
- Objects: OWLv2-large boxes from a text prompt per object, then a SAM 2.1-large mask per box, on a remote GPU (`scripts/sam_server.py`, `cap_so101/sam.py`). The colour detector below is not used.

"""

METHOD = """\
### Perception
- Objects: OpenCV colour and texture thresholds on the C270 frame, no learned model (`cap_so101/detect.py`).
- Mouse: the largest blue patch that is not long and thin. Pen: every other blue patch. Remote and arm: dark pixels. Cushion: Laplacian texture.
- Table: homography from image pixels to table centimetres, fitted from 7 points the gripper touched (`so101/table_homography.json`, leave-one-out error 0.44 cm mean).
- The mouse stands about 4 cm tall, so only its lowest pixel per column, where it meets the table, is mapped.

### Arm
- Joints read back from the Feetech bus, drawn on `so101/so101.urdf` through `so101/urdf_map.json`.
- Motion: inverse kinematics on the MuJoCo model (`cap_so101/kinematics.py`). Moves rise to 8 cm, cross, then descend.
- Grasp: wrist roll turned so the jaws open across the pen's axis, measured from the camera.

### MuJoCo recreation
- The arm model at the read-back joints, with boxes and capsules at the objects' table positions. Heights and the cat toy's size are by eye.

### Push model
- Predicted displacement = gain x gripper travel after contact. The gain starts at 1.
- The gain is refitted by least squares through the origin after every push, failed pushes included. Each push's prediction and measurement are in the log.
- Pushes go through the mouse's estimated centre, from its measured side and its 10.5 by 7 cm size.
"""


def blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="camera", name="Camera"),
                rrb.TextDocumentView(origin="plan/code", name="Code"),
                row_shares=[3, 2],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(origin="sim", name="MuJoCo"),
                rrb.Spatial3DView(origin="/", name="Arm", contents=["/so101/**", "/tf_static/**"]),
            ),
            rrb.Vertical(
                rrb.Tabs(
                    rrb.TextDocumentView(origin="plan/subtasks", name="Plan"),
                    rrb.TextDocumentView(origin="plan/method", name="Method"),
                ),
                rrb.TextLogView(origin="plan/log", name="Log"),
                row_shares=[3, 2],
            ),
            column_shares=[5, 4, 3],
        ),
        collapse_panels=True,
    )


class SimArm:
    """Stands in for Arm in a dry run. Moves land on the clamped target at once, no motor is touched."""

    def __init__(self, start: np.ndarray, limits: dict[str, list[float]]) -> None:
        self.joints = np.asarray(start, dtype=np.float64).copy()
        self.limits = limits

    def read(self) -> np.ndarray:
        return self.joints.copy()

    def set_torque(self, enabled: bool) -> None:
        pass

    def move(self, target: np.ndarray) -> np.ndarray:
        return self.follow([target])

    def follow(self, waypoints: list[np.ndarray]) -> np.ndarray:
        lower = np.array([self.limits[name][0] for name in JOINT_NAMES])
        upper = np.array([self.limits[name][1] for name in JOINT_NAMES])
        for target in waypoints:
            self.joints = np.clip(np.asarray(target, dtype=np.float64), lower, upper)
        return self.read()

    def close(self) -> None:
        pass


class Demo:
    def __init__(self, run_dir: Path, goal: str, subtasks: list[str], dry: bool, live: bool,
                 screenshots: bool = False, use_sam: bool = False) -> None:
        self.use_sam = use_sam
        self.run_dir = run_dir
        self.dry = dry
        self.screenshots = screenshots
        (run_dir / "frames").mkdir(parents=True)
        self.state = {"goal": goal, "dry_run": dry, "subtasks": [{"text": text, "done": False} for text in subtasks],
                      "pushes": [], "gain": 1.0}
        self._save_state()

        rr.init(APP_ID + ("-dry-run" if dry else ""))
        path = run_dir / "demo.rrd"
        if live:
            # Also stream to a viewer already listening on the default port (`rerun`).
            rr.set_sinks(rr.GrpcSink(), rr.FileSink(str(path)), default_blueprint=blueprint())
        else:
            rr.save(path, default_blueprint=blueprint())
        self._now()
        tree = UrdfTree.from_file_path(str(URDF))
        tree.log_urdf_to_recording()
        self.robot_name = tree.name
        self.urdf_joints = {joint.name: joint for joint in tree.joints()}
        self.urdf_map = json.loads(URDF_MAP.read_text())
        method = METHOD_SAM + METHOD if use_sam else METHOD
        rr.log("plan/method", rr.TextDocument(method, media_type=rr.MediaType.MARKDOWN), static=True)
        rr.log("model/predicted_cm", rr.SeriesPoints(colors=[COLOURS["predicted"]], names=["predicted"], marker_sizes=[6]), static=True)
        rr.log("model/measured_cm", rr.SeriesPoints(colors=[COLOURS["measured"]], names=["measured"], marker_sizes=[6]), static=True)

        self.kinematics = Kinematics()
        self.recreation = Recreation(self.kinematics)
        self.homography = load_homography()
        self.camera = Camera()
        real_arm = Arm()
        if real_arm.limits is None:
            raise RuntimeError("so101/joint_limits.json is missing.")
        if dry:
            # Read where the arm is, then let go of the bus: no motor command in a dry run.
            self.arm = SimArm(real_arm.read(), real_arm.limits)
            real_arm.close()
        else:
            self.arm = real_arm
        self.desk: Desk | None = None
        self.last_frame: str | None = None
        self.last_sim: str | None = None
        if dry:
            self.log("DRY RUN, no motion")
        if goal:
            self.plan(goal, subtasks)

    # Bookkeeping

    def _now(self) -> None:
        rr.set_time("time", timestamp=time.time())

    def _save_state(self) -> None:
        (self.run_dir / "state.json").write_text(json.dumps(self.state, indent=2) + "\n")

    def log(self, text: str, level: str = rr.TextLogLevel.INFO) -> None:
        self._now()
        rr.log("plan/log", rr.TextLog(text, level=level))
        print(text)

    def close(self) -> None:
        self._save_state()
        rr.disconnect()
        self.camera.close()
        self.recreation.close()
        self.arm.close()

    # Plan and code

    def log_plan(self) -> None:
        self._now()
        heading = ("DRY RUN: " if self.dry else "") + self.state["goal"]
        lines = [f"## {heading}", ""]
        for index, subtask in enumerate(self.state["subtasks"]):
            lines.append(f"- [{'x' if subtask['done'] else ' '}] {index}. {subtask['text']}")
        rr.log("plan/subtasks", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN))

    def plan(self, goal: str, subtasks: list[str]) -> None:
        """Write the goal and the subtask list before acting."""
        self.state["goal"] = goal
        self.state["subtasks"] = [{"text": text, "done": False} for text in subtasks]
        self._save_state()
        self.log(f"goal: {goal}")
        self.log_plan()

    def log_code(self, heading: str, code: str) -> None:
        self._now()
        rr.log("plan/code", rr.TextDocument(f"### {heading}\n\n```python\n{code}\n```", media_type=rr.MediaType.MARKDOWN))
        rr.log("plan/code_log", rr.TextLog(f"{heading}:\n{code}"))
        self.log(f"running {heading}")

    def mark_done(self, index: int, evidence: str) -> None:
        """Tick a subtask. Call only after the camera has confirmed it, and say how."""
        self.state["subtasks"][index]["done"] = True
        self.state["subtasks"][index]["evidence"] = evidence
        self._save_state()
        self.log(f"subtask {index} done: {evidence}")
        self.log_plan()

    # Camera and desk

    def look(self) -> dict[str, detect.Detection]:
        """Grab a frame, find the objects, and log both."""
        frame = self.camera.snap()
        found = sam.detect(frame) if self.use_sam else detect.detect(frame)
        self._now()
        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("could not encode the frame")
        path = self.run_dir / "frames" / f"{datetime.now():%H%M%S%f}.jpg"
        path.write_bytes(jpeg.tobytes())
        self.last_frame = str(path)
        rr.log("camera/image", rr.EncodedImage(contents=jpeg.tobytes(), media_type="image/jpeg"))
        names = [name for name in found if name in COLOURS]
        rr.log("camera/objects", rr.Boxes2D(
            array=[found[name].box for name in names], array_format=rr.Box2DFormat.XYXY,
            labels=names, colors=[COLOURS[name] for name in names]))
        return found

    def see(self) -> tuple[Desk, dict[str, detect.Detection]]:
        """The desk in table cm. In a dry run, the first look sets it and the task's expectations move it."""
        found = self.look()
        if self.desk is None or not self.dry:
            held = self.desk.pen_held if self.desk is not None else False
            self.desk = Desk.from_detections(found, self.homography)
            self.desk.pen_held = held
        self.render()
        return self.desk, found

    def expect(self, **changes: object) -> None:
        """What a step should have done to the desk. A dry run applies it, a real run only draws the held pen."""
        if self.desk is None:
            raise RuntimeError("expect before the first look")
        for name, value in changes.items():
            if self.dry or name == "pen_held":
                setattr(self.desk, name, value)
        self.render()

    def expect_push(self, direction: np.ndarray, travel_cm: float) -> None:
        """A dry run moves the simulated mouse DRY_PUSH_GAIN times the travel after contact. A real run does nothing."""
        if self.dry:
            self.desk.mouse_front = self.desk.mouse_front + np.asarray(direction) * travel_cm * DRY_PUSH_GAIN
            self.log(f"dry run: simulated mouse moved {travel_cm * DRY_PUSH_GAIN:.1f} cm, "
                     f"{DRY_PUSH_GAIN} times the travel after contact")
            self.render()

    def overlay(self, path_cm: list[np.ndarray] | None = None, ghost: tuple[float, float, float, float] | None = None) -> None:
        """Draw a planned path given in table cm and a predicted end box in pixels, or clear them."""
        self._now()
        if path_cm:
            pixels = to_pixel(self.homography, np.array(path_cm, dtype=np.float64))
            rr.log("camera/plan_path", rr.LineStrips2D([pixels], colors=[COLOURS["path"]], radii=3))
        else:
            rr.log("camera/plan_path", rr.Clear(recursive=False))
        if ghost:
            rr.log("camera/predicted_end", rr.Boxes2D(array=[ghost], array_format=rr.Box2DFormat.XYXY,
                                                      labels=["predicted"], colors=[(*COLOURS["predicted"], 140)]))
        else:
            rr.log("camera/predicted_end", rr.Clear(recursive=False))

    def to_table(self, pixels: np.ndarray) -> np.ndarray:
        return to_table(self.homography, np.atleast_2d(np.asarray(pixels, dtype=np.float64)))

    def to_pixel(self, points_cm: np.ndarray) -> np.ndarray:
        return to_pixel(self.homography, np.atleast_2d(np.asarray(points_cm, dtype=np.float64)))

    # Arm

    def render(self) -> None:
        if self.desk is None:
            return
        self._now()
        image = self.recreation.render(self.arm.read(), self.desk)
        rr.log("sim/image", rr.Image(image))
        path = self.run_dir / "sim" / f"{datetime.now():%H%M%S%f}.png"
        path.parent.mkdir(exist_ok=True)
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"could not write {path}")
        self.last_sim = str(path)

    def screenshot(self, label: str) -> None:
        """Grab the whole screen with grim, so the Rerun window is kept as seen. Off unless --screenshots."""
        if not self.screenshots:
            return
        path = self.run_dir / "screens" / f"{datetime.now():%H%M%S}-{label}.png"
        path.parent.mkdir(exist_ok=True)
        subprocess.run(["grim", str(path)], check=True)

    def log_arm(self, joints: np.ndarray) -> None:
        names, angles = [], []
        for bus_name, value in zip(JOINT_NAMES, joints):
            entry = self.urdf_map[bus_name]
            names.append(entry["urdf"])
            angles.append(np.deg2rad(entry["sign"] * entry["scale"] * value + entry["offset_deg"]))
        self._now()
        # No clamping: the real arm goes past the URDF limits (Pitch at rest), and
        # clamping would also hide a wrong sign or offset.
        log_joint_angles(self.robot_name, self.urdf_joints, names, angles, clamp=False)
        self.render()

    def move(self, target: np.ndarray) -> np.ndarray:
        """One straight move in joint space, refused if it dips below both of its ends."""
        after = checked_move(self.arm, self.kinematics, target)
        self.log_arm(after)
        return after

    def follow(self, waypoints: list[np.ndarray]) -> np.ndarray:
        """Through all waypoints without stopping, every segment checked first."""
        after = checked_follow(self.arm, self.kinematics, waypoints)
        self.log_arm(after)
        return after

    def travel(self, final: np.ndarray) -> np.ndarray:
        """Up to 8 cm, across at that height, then down to final."""
        for after in travel(self.arm, self.kinematics, final):
            self.log_arm(after)
        return after

    def grip(self, percent: float) -> np.ndarray:
        """Open or close the gripper, everything else held."""
        target = self.arm.read()
        target[JOINT_NAMES.index("gripper")] = percent
        return self.move(target)

    # Push model

    def predict(self, travel_cm: float) -> float:
        return self.state["gain"] * travel_cm

    def record_push(self, travel_cm: float, predicted_cm: float, measured_cm: float) -> None:
        """Log one push and refit the gain on all pushes so far, failed ones included."""
        pushes = self.state["pushes"]
        pushes.append({"travel_cm": travel_cm, "predicted_cm": predicted_cm, "measured_cm": measured_cm})
        travel_all = np.array([push["travel_cm"] for push in pushes])
        measured_all = np.array([push["measured_cm"] for push in pushes])
        if (travel_all ** 2).sum() > 0:
            self.state["gain"] = float((travel_all * measured_all).sum() / (travel_all ** 2).sum())
        self._save_state()
        self._now()
        rr.set_time("push", sequence=len(pushes))
        rr.log("model/predicted_cm", rr.Scalars(predicted_cm))
        rr.log("model/measured_cm", rr.Scalars(measured_cm))
        rr.reset_time()
        self.log(f"push {len(pushes)}: travel after contact {travel_cm:.1f} cm, predicted {predicted_cm:.1f} cm, "
                 f"measured {measured_cm:.1f} cm, gain now {self.state['gain']:.2f}")
        self.screenshot(f"push{len(pushes)}")


def run_step(code: str, scope: dict) -> dict:
    """Run one step's code like a REPL cell: the last line's value comes back if it is an expression."""
    started = time.monotonic()
    output = io.StringIO()
    result: dict = {"ok": True, "value": None, "error": None}
    try:
        tree = ast.parse(code)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        with contextlib.redirect_stdout(output):
            exec(compile(tree, "<step>", "exec"), scope)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), "<step>", "eval"), scope)
                result["value"] = repr(value) if value is not None else None
    except Exception:
        result["ok"] = False
        result["error"] = traceback.format_exc(limit=4)
    result["output"] = output.getvalue()
    result["seconds"] = round(time.monotonic() - started, 1)
    return result


def serve(args: argparse.Namespace) -> None:
    """Hold the arm, camera, model and Rerun stream open and run steps sent by scripts/act.py."""
    run_dir = REPO / "runs" / "demo" / (f"{datetime.now():%Y%m%d-%H%M%S}-live" + ("-dry-run" if args.dry_run else ""))
    demo = Demo(run_dir, "", [], args.dry_run, args.live, args.screenshots, args.sam)
    scope = {"demo": demo, "np": np, "skills": skills}
    for name in ("look", "home", "pick_pen", "place_pen", "push_mouse", "point_at_remote"):
        scope[name] = functools.partial(getattr(skills, name), demo)
    SOCKET.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(SOCKET))
    server.listen(1)
    try:
        demo.see()
        demo.log_arm(demo.arm.read())
        print(f"serving {run_dir} on {SOCKET}", flush=True)
        step = 0
        while True:
            connection, _ = server.accept()
            with connection, connection.makefile("rwb") as stream:
                request = json.loads(stream.readline())
                if request.get("shutdown"):
                    stream.write(b'{"ok": true, "value": "shutting down"}\n')
                    stream.flush()
                    break
                step += 1
                demo.log_code(f"step {step}", request["code"])
                result = run_step(request["code"], scope)
                if not result["ok"]:
                    demo.log(f"step {step} failed: {result['error'].strip().splitlines()[-1]}", level=rr.TextLogLevel.ERROR)
                    demo.screenshot(f"step{step}-error")
                else:
                    demo.screenshot(f"step{step}")
                result.update(step=step, frame=demo.last_frame, sim=demo.last_sim)
                stream.write((json.dumps(result) + "\n").encode())
                stream.flush()
    finally:
        server.close()
        SOCKET.unlink(missing_ok=True)
        demo.close()
    print(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="also stream to a running `rerun` viewer")
    commands = parser.add_subparsers(dest="command", required=True)
    serve_parser = commands.add_parser("serve")
    serve_parser.add_argument("--dry-run", action="store_true")
    serve_parser.add_argument("--screenshots", action="store_true", help="grab the screen with grim after each step")
    serve_parser.add_argument("--sam", action="store_true", help="masks from scripts/sam_server.py through the tunnel")
    args = parser.parse_args()
    serve(args)


if __name__ == "__main__":
    main()
