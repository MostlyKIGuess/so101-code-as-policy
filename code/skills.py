"""Skills the live agent calls on the arm, one at a time.

Each takes the robot context (Demo in scripts/demo.py), checks its own outcome with the
camera where it can, and returns what it measured. It raises instead of guessing when an
input is missing or a check fails. What to call next (push again, retry, give up) is the
caller's decision, made from what the last call returned. Positions are table cm in the
model frame of cap_so101/kinematics.py: -y towards the camera, the table at z = 0.

The numbers below come from the 24 Oct runs, cited where they are set.
"""

import numpy as np

from cap_so101.arm import JOINT_NAMES
from cap_so101.kinematics import REST, TRAVEL_CM, TRAVEL_FLOOR_CM

GRIPPER = JOINT_NAMES.index("gripper")
ROLL = JOINT_NAMES.index("wrist_roll")
OPEN = 15.0       # jaw tips 3.1 cm apart in the model, the pen is about 1.1 cm thick
CLOSED = 1.3      # the gripper's recorded lower limit, the pen stops it before that
HOME_GRIPPER = 3.4  # the gripper's reading at rest on 24 Oct
# Wrist roll at rest before the pen run on 24 Oct. Going home restores it, since the tip
# sits 0.9 cm off the roll axis and a turned roll left the pointing pose 2 degrees short.
HOME_ROLL = 2.1
SIDE_CM = 0.9     # fixed jaw tip beside the pen's middle: radius 0.55 plus 0.35 clearance
GRASP_CM = 0.5    # tip height for the grasp, the read-back tip reached 0.3 to 0.5 on the touches
# The pen hung tilted in the grip (runs/demo/20261024-222227/frames/222310395265.jpg), so it is
# carried as high as a jaw-down pose reaches inside the recorded limits, up to 10 cm.
CARRY_CM = 10.0
DROP_CM = 3.0
# Open table on the arm's right, behind the cushion, where touches 4 and 5 were made.
PLACE_CENTRE = np.array([-14.0, -12.0])
PLACE_AXIS = np.array([1.0, 0.0])
PEN_HALF_CM = 6.5  # half of the pen's 13 to 14 cm length measured through the homography
PEN_MOVED_PX = 40  # a lifted pen moved 133 px in the frame on 24 Oct (run 20261024-225435)

PUSH_HEIGHT_CM = 1.5      # low on the mouse's side, which stands about 4 cm
# Gap between the jaws and the mouse before the push starts. 3 cm put the start at 24.9 cm
# out on the arm's left, where a dry run found no jaw-down pose at the push's wrist roll.
PUSH_CLEARANCE_CM = 2.0
PUSH_STEP_CM = 2.0        # the push is a straight line solved every 2 cm, run without stopping
# Jaw leans tried in order. Straight down could not reach the mouse's far side at 23.9 cm
# out above 1.5 cm, 10 degrees reached it from 1.5 to 8 cm (scan, 24 Oct).
PUSH_TILTS_DEG = (0.0, 10.0, 20.0, 30.0, 40.0)
# The mouse's long and short sides. 10.5 cm measured lying sideways and 6.9 to 7.1 cm
# measured front-on, both through the homography (run 20261024-230407-live). The side that
# faces the camera is measured, the other is taken from these. A push off the centre spun
# the mouse 90 degrees on 24 Oct, so the push line goes through this estimated centre.
MOUSE_SIZE_CM = (10.5, 7.0)
# Pushes tried when the straight line to the cushion has no reachable contact point, in
# degrees off that line. On 24 Oct the contact point behind the mouse was past the pan
# limit and a push 15 to 20 degrees flatter was reachable.
PUSH_TURNS_DEG = (0.0, 15.0, -15.0, 30.0, -30.0)
TOUCH_GAP_CM = 1.0        # about two leave-one-out homography errors (0.44 cm mean)
AIM_SHORT_CM = 0.5        # a push aims this short of the cushion so it does not shove the cat

# Where the tip stops: cm short of the remote and cm up, tried in order. A scan on 24 Oct
# found poses inside the recorded limits for 4 to 10 cm back and 4 to 12 cm up.
POINT_TRIES = ((8.0, 8.0), (6.0, 8.0), (8.0, 10.0), (6.0, 10.0), (4.0, 8.0))
POINT_HIT_CM = 2.0        # the pointing ray must meet the table this close to the remote
# Where the gripper waits while the camera measures: 10 cm over the open table on the arm's
# right, out of the frame's view of the mouse, cushion and remote (24 Oct).
PARK_CM = np.array([-14.0, -8.0, 10.0])


def _round(vector: np.ndarray) -> list[float]:
    return np.round(np.asarray(vector, dtype=np.float64), 1).tolist()


def home(demo) -> dict:
    """Back to the folded rest pose: up, across, down."""
    target = demo.arm.read()
    for name, value in REST.items():
        target[JOINT_NAMES.index(name)] = value
    target[GRIPPER] = HOME_GRIPPER
    target[ROLL] = HOME_ROLL
    after = demo.travel(target)
    return {"joints": _round(after)}


def park(demo) -> dict:
    """Hold the gripper at PARK_CM, away from the objects, so a look sees them whole.

    At home the gripper hangs where the first push left the mouse, and their dark shapes
    merged in the frame (runs/demo/20261024-230407-live/detect-after-fix.jpg). Parked,
    the same mouse was found whole (detect-parked.jpg).
    """
    target = demo.kinematics.solve_tip_down(PARK_CM, demo.arm.read(), demo.arm.limits)
    after = demo.travel(target)
    return {"joints": _round(after)}


def mouse_footprint(desk) -> tuple[np.ndarray, float, float]:
    """Estimated centre, width along x and depth along y of the mouse on the table, in cm."""
    if desk.mouse_front is None:
        raise RuntimeError("no mouse in the frame")
    left, right, _ = desk.mouse_edges()
    width = right - left
    long_side, short_side = MOUSE_SIZE_CM
    depth = float(np.clip(long_side + short_side - width, short_side, long_side))
    front = float(desk.mouse_front[:, 1].min())
    return np.array([(left + right) / 2, front + depth / 2]), width, depth


def half_along(direction: np.ndarray, width: float, depth: float) -> float:
    """How far the mouse's footprint box reaches from its centre along a direction."""
    return abs(float(direction[0])) * width / 2 + abs(float(direction[1])) * depth / 2


def outline_gap(desk) -> float:
    """Shortest distance from where the mouse meets the table to the cushion's footprint.

    The earlier gap, centre to cushion less half the width, said touching at a real
    1.6 cm once the cushion lay in front of the mouse rather than beside it (24 Oct).
    """
    if desk.mouse_front is None or desk.cushion is None:
        raise RuntimeError("mouse or cushion not in the frame")
    return min(float(np.linalg.norm(point - desk.cushion.nearest(point))) for point in desk.mouse_front)


def push_line(desk) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Mouse centre, its reach along the push, unit direction to the cushion, and the outline gap."""
    if desk.cushion is None:
        raise RuntimeError("no cushion in the frame")
    centre, width, depth = mouse_footprint(desk)
    target = desk.cushion.nearest(centre)
    direction = (target - centre) / np.linalg.norm(target - centre)
    return centre, half_along(direction, width, depth), direction, outline_gap(desk)


def look(demo) -> dict:
    """One fresh frame: where each object is on the table, and the mouse's gap to the cushion."""
    desk, found = demo.see()
    seen = {"objects": sorted(name for name in found if name != "arm")}
    if desk.pen is not None:
        seen["pen_ends_cm"] = [_round(end) for end in desk.pen]
    if desk.mouse_front is not None:
        left, right, _ = desk.mouse_edges()
        seen["mouse_cm"] = {"left": round(left, 1), "right": round(right, 1),
                            "front": round(float(desk.mouse_front[:, 1].min()), 1)}
    if desk.cushion is not None:
        seen["cushion_cm"] = {"centre": _round(desk.cushion.centre), "half": _round(desk.cushion.half)}
    if desk.remote is not None:
        seen["remote_cm"] = {"centre": _round(desk.remote.centre), "half": _round(desk.remote.half)}
    if desk.mouse_front is not None and desk.cushion is not None:
        seen["mouse_gap_to_cushion_cm"] = round(push_line(desk)[3], 1)
    return seen


def highest_grasp(kinematics, centre: np.ndarray, axis: np.ndarray, seed: np.ndarray, limits) -> np.ndarray:
    """The highest jaw-down grasp pose from CARRY_CM down to TRAVEL_CM, in 0.5 cm steps."""
    for height in np.arange(CARRY_CM, TRAVEL_CM - 0.01, -0.5):
        try:
            return kinematics.solve_grasp(centre, axis, height, SIDE_CM, seed, limits)
        except ValueError:
            continue
    raise ValueError(f"no jaw-down pose at {TRAVEL_CM} cm or above over {_round(centre)}")


def pick_pen(demo) -> dict:
    """Grasp the pen across its middle with the jaws 15 percent open, close, and lift it as high as reach allows.

    Checked by the camera: the pen must move off its place in the frame.
    """
    desk, found = demo.see()
    if desk.pen is None:
        raise RuntimeError("no pen in the frame")
    end_a, end_b = desk.pen
    centre, axis = (end_a + end_b) / 2, end_b - end_a
    kinematics, limits = demo.kinematics, demo.arm.limits
    seed = demo.arm.read()
    seed[GRIPPER] = OPEN
    above = kinematics.solve_grasp(centre, axis, TRAVEL_CM, SIDE_CM, seed, limits)
    low = kinematics.solve_grasp(centre, axis, 3.0, SIDE_CM, above, limits)
    grasp = kinematics.solve_grasp(centre, axis, GRASP_CM, SIDE_CM, low, limits)
    carry = highest_grasp(kinematics, centre, axis, grasp, limits)
    demo.overlay(path_cm=[kinematics.tip(grasp)[:2], kinematics.moving_tip(grasp)[:2]])  # where the jaws close
    demo.log(f"pick pen at {_round(centre)} cm, carry at {kinematics.tip(carry)[2]:.1f} cm")

    demo.travel(above)
    demo.move(low)
    demo.move(grasp)
    held = float(demo.grip(CLOSED)[GRIPPER])
    for target in (low, carry):
        lift = target.copy()
        lift[GRIPPER] = CLOSED
        demo.move(lift)
    demo.expect(pen_held=True)
    _, found_after = demo.see()
    result = {"grasp_cm": _round(centre), "carry_cm": round(float(kinematics.tip(carry)[2]), 1),
              "gripper_percent": round(held, 1)}
    if demo.dry:
        result["check"] = "dry run, no camera check"
        return result
    if "pen" not in found_after:
        raise RuntimeError("no pen in the frame after lifting")
    moved = float(np.linalg.norm(np.subtract(found_after["pen"].centre, found["pen"].centre)))
    result["pen_moved_px"] = round(moved)
    if moved < PEN_MOVED_PX:
        raise RuntimeError(f"pen moved only {moved:.0f} px, it is still on the table")
    return result


def place_pen(demo, centre: np.ndarray = PLACE_CENTRE, axis: np.ndarray = PLACE_AXIS) -> dict:
    """Carry the held pen to centre, turn it along axis, drop it from 3 cm, go home, and look where it lies."""
    centre, axis = np.asarray(centre, dtype=np.float64), np.asarray(axis, dtype=np.float64)
    kinematics, limits = demo.kinematics, demo.arm.limits
    carry = highest_grasp(kinematics, centre, axis, demo.arm.read(), limits)
    drop = kinematics.solve_grasp(centre, axis, DROP_CM, SIDE_CM, carry, limits)
    unit = axis / np.linalg.norm(axis)
    spot = (centre - PEN_HALF_CM * unit, centre + PEN_HALF_CM * unit)
    demo.overlay(path_cm=list(spot))

    demo.move(carry)
    demo.move(drop)
    demo.grip(OPEN)
    demo.expect(pen_held=False, pen=spot)
    up = carry.copy()
    up[GRIPPER] = OPEN
    demo.move(up)
    home(demo)

    desk, _ = demo.see()
    if desk.pen is None:
        raise RuntimeError("no pen in the frame after placing")
    result = {"pen_ends_cm": [_round(end) for end in desk.pen],
              "from_target_cm": round(float(np.linalg.norm((desk.pen[0] + desk.pen[1]) / 2 - centre)), 1)}
    if desk.mouse_front is not None and desk.cushion is not None:
        line_centre, half, direction, gap = push_line(desk)
        line_end = line_centre + direction * (gap + half)
        span = line_end - line_centre

        def distance(point: np.ndarray) -> float:
            along = np.clip((point - line_centre) @ span / (span @ span), 0, 1)
            return float(np.linalg.norm(point - (line_centre + along * span)))

        points = [desk.pen[0] + (desk.pen[1] - desk.pen[0]) * step / 20 for step in range(21)]
        result["clearance_to_mouse_push_line_cm"] = round(min(distance(point) for point in points), 1)
        result["mouse_half_width_cm"] = round(half, 1)
        demo.overlay(path_cm=[line_centre, line_end])
    return result


def plan_push(kinematics, limits, start: np.ndarray, end: np.ndarray, direction: np.ndarray,
              seed: np.ndarray) -> tuple[float, np.ndarray, list[np.ndarray], np.ndarray]:
    """Approach, push line and retreat poses at the smallest jaw lean that reaches all of them.

    The jaw faces the push: its opening runs along the push direction. The approach is
    as high above the start as reach allows, from TRAVEL_CM down to TRAVEL_FLOOR_CM.
    """
    across = np.array([-direction[1], direction[0]])
    steps = max(1, int(np.ceil(np.linalg.norm(end - start) / PUSH_STEP_CM)))
    for tilt in PUSH_TILTS_DEG:
        try:
            approach = None
            for height in np.arange(TRAVEL_CM, TRAVEL_FLOOR_CM - 0.01, -0.5):
                try:
                    approach = kinematics.solve_grasp(start, across, height, 0.0, seed, limits, tilt_deg=tilt)
                    break
                except ValueError:
                    continue
            if approach is None:
                continue
            poses, current = [], approach
            for step in range(steps + 1):
                point = start + (end - start) * step / steps
                current = kinematics.solve_grasp(point, across, PUSH_HEIGHT_CM, 0.0, current, limits, tilt_deg=tilt)
                poses.append(current)
            back = kinematics.solve_grasp(end - direction * PUSH_CLEARANCE_CM, across, PUSH_HEIGHT_CM, 0.0,
                                          current, limits, tilt_deg=tilt)
            return tilt, approach, poses, back
        except ValueError:
            continue
    raise ValueError(f"no jaw lean in {PUSH_TILTS_DEG} reaches the push from {_round(start)} to {_round(end)}")


def push_mouse(demo, aim_cm: float | None = None, direction: np.ndarray | None = None) -> dict:
    """One push of the mouse, predicted before and measured after.

    By default along the line from the mouse's estimated centre to the nearest point of
    the cushion, aiming the outline gap less AIM_SHORT_CM. If that push has no reachable
    contact point, lines turned by PUSH_TURNS_DEG are tried. `direction` forces a line and
    then needs aim_cm. Returns the gap before and after, so the caller decides whether to
    push again. Does not move when the mouse is already within TOUCH_GAP_CM.
    """
    kinematics, limits = demo.kinematics, demo.arm.limits
    desk, found = demo.see()
    before = desk.mouse_front.mean(axis=0).copy() if desk.mouse_front is not None else None  # a dry run moves this desk in place
    centre, _, line, gap = push_line(desk)
    _, width, depth = mouse_footprint(desk)
    if gap <= TOUCH_GAP_CM:
        return {"gap_before_cm": round(gap, 1), "touching": True, "pushed": False}
    if direction is not None:
        if aim_cm is None:
            raise ValueError("a push along a given direction needs aim_cm")
        candidates = [(0.0, np.asarray(direction, dtype=np.float64) / np.linalg.norm(direction))]
    else:
        candidates = []
        for turn in PUSH_TURNS_DEG:
            angle = np.deg2rad(turn)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            candidates.append((turn, rotation @ line))
    aim = aim_cm if aim_cm is not None else (gap - AIM_SHORT_CM if gap - AIM_SHORT_CM > 0.5 else gap)
    travel_cm = aim / demo.state["gain"]
    predicted = demo.predict(travel_cm)
    for turn, direction in candidates:
        half = half_along(direction, width, depth)
        start = centre - direction * (half + PUSH_CLEARANCE_CM)
        end = centre - direction * half + direction * travel_cm
        try:
            tilt, approach, poses, back = plan_push(kinematics, limits, start, end, direction, demo.arm.read())
            break
        except ValueError:
            demo.log(f"no reachable push {turn:+.0f} degrees off the line to the cushion")
    else:
        raise ValueError(f"no reachable push within {PUSH_TURNS_DEG} degrees of the line to the cushion")
    shift = demo.to_pixel(centre + direction * predicted)[0] - demo.to_pixel(centre)[0]
    x0, y0, x1, y1 = found["mouse"].box
    demo.overlay(path_cm=[start, end], ghost=(x0 + shift[0], y0 + shift[1], x1 + shift[0], y1 + shift[1]))
    demo.log(f"predict the mouse moves {predicted:.1f} cm (gain {demo.state['gain']:.2f} x {travel_cm:.1f} cm "
             f"after contact), {turn:+.0f} degrees off the line, jaw leaned {tilt:.0f} degrees")

    demo.travel(approach)
    demo.move(poses[0])
    demo.follow(poses[1:] + [back])
    demo.expect_push(direction, travel_cm)
    park(demo)

    after, _ = demo.see()
    if after.mouse_front is None:
        raise RuntimeError("no mouse in the frame after the push")
    measured = float((after.mouse_front.mean(axis=0) - before) @ direction)
    demo.record_push(travel_cm, predicted, measured)
    gap_after = outline_gap(after)
    return {"gap_before_cm": round(gap, 1), "predicted_cm": round(predicted, 1), "measured_cm": round(measured, 1),
            "gap_after_cm": round(gap_after, 1), "touching": gap_after <= TOUCH_GAP_CM, "tilt_deg": tilt,
            "turn_deg": turn, "gain_now": round(demo.state["gain"], 2), "pushed": True}


def point_at_remote(demo) -> dict:
    """Hold the tip short of the remote and above it, with the jaw pointing at it, then check the ray."""
    desk, _ = demo.see()
    if desk.remote is None:
        raise RuntimeError("no remote in the frame")
    remote = desk.remote.centre
    kinematics, limits = demo.kinematics, demo.arm.limits
    # The arm points only within a vertical plane through its pan axis, so aim from there.
    toward = (remote - kinematics.pan_axis_cm) / np.linalg.norm(remote - kinematics.pan_axis_cm)
    pose = None
    for back, height in POINT_TRIES:
        tip_target = np.r_[remote - toward * back, height]
        try:
            pose = kinematics.solve_pose(tip_target, np.r_[remote, 1.0] - tip_target, demo.arm.read(), limits)
            break
        except ValueError:
            continue
    if pose is None:
        raise ValueError(f"no pointing pose at the remote {_round(remote)} from {POINT_TRIES}")
    demo.overlay(path_cm=[tip_target[:2], remote])
    demo.travel(pose)
    demo.see()

    after = demo.arm.read()
    tip, jaw = kinematics.tip(after), kinematics.jaw_direction(after)
    if jaw[2] >= 0:
        raise RuntimeError("the jaw does not point down towards the table")
    hit = (tip + jaw * (-tip[2] / jaw[2]))[:2]
    miss = float(np.linalg.norm(hit - desk.remote.nearest(hit)))
    if miss > POINT_HIT_CM:
        raise RuntimeError(f"pointing ray misses the remote by {miss:.1f} cm")
    return {"remote_cm": _round(remote), "ray_hits_table_cm": _round(hit), "miss_cm": round(miss, 1),
            "from": {"back_cm": back, "height_cm": height}}
