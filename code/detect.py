"""Find the desk objects in a C270 frame with colour and texture thresholds.

No learned model. The mouse and the pen are the only blue things on the desk,
told apart by shape (the pen is long and thin), and each takes the dark patches
touching it. The arm is the dark patch that crosses the table edge
(TABLE_TOP_Y), joined there to the dark room behind the desk. The remote is the
largest dark patch left on the table. The cushion under the cat toy is the
fine texture that belongs to none of those.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Table edge in the frame, read off runs/after-limits.jpg (24 Oct). The camera
# is clipped to the partition, so this holds until someone moves it.
TABLE_TOP_Y = 100
# The pen's upper barrel is dark navy (value 16 to 33, saturation 255 in
# runs/pen-plan.jpg). A value floor of 15 keeps it blue without catching the black arm.
BLUE_LOW, BLUE_HIGH = (100, 150, 15), (130, 255, 255)
DARK_VALUE_MAX = 60
TEXTURE_BLUR = 41
TEXTURE_MIN = 7.0
MIN_AREA = 500
# Long side over short side of the blue patch. On the 24 Oct frames the pen
# was 6.0 to 6.4 and the mouse 1.2 to 1.4.
PEN_ELONGATION = 3.0
# Patches closer than this belong together, e.g. the pen's cap and barrel.
TOUCH_PX = 9
# How far the mouse's black body reaches past its blue shell, in shell heights below and
# shell widths to each side. Measured 1.0 and 0.24 on
# runs/demo/20261024-230407-live/frames/230551521305.jpg, with some margin.
MOUSE_BODY_BELOW = 1.2
MOUSE_BODY_SIDE = 0.3


@dataclass
class Detection:
    name: str
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels
    centre: tuple[float, float]
    mask: np.ndarray  # bool, frame height by width


def _components(mask: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Connected patches of a mask, largest first, split into those on the table and those crossing its edge."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    on_table, crossing = [], []
    for index in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1:
        if stats[index, cv2.CC_STAT_AREA] < MIN_AREA:
            break
        (crossing if stats[index, cv2.CC_STAT_TOP] < TABLE_TOP_Y else on_table).append(labels == index)
    return on_table, crossing


def _detection(name: str, mask: np.ndarray) -> Detection:
    ys, xs = np.nonzero(mask)
    return Detection(name, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                     (float(xs.mean()), float(ys.mean())), mask)


def _elongation(mask: np.ndarray) -> float:
    points = np.column_stack(np.nonzero(mask)).astype(np.float32)
    (_, _), (width, height), _ = cv2.minAreaRect(points)
    return max(width, height) / max(1.0, min(width, height))


def _grow(mask: np.ndarray) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), np.ones((TOUCH_PX, TOUCH_PX), np.uint8)).astype(bool)


def detect(frame: np.ndarray) -> dict[str, Detection]:
    """Return the objects found in a BGR frame, by name. Missing objects are absent."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    found: dict[str, Detection] = {}

    # The mouse is the largest blue patch that is not long and thin. Every other blue
    # patch is pen: its clear grip splits it into two pieces.
    pen_pieces = []
    for patch in _components(cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH))[0]:
        if "mouse" not in found and _elongation(patch) <= PEN_ELONGATION:
            found["mouse"] = _detection("mouse", patch)
        else:
            pen_pieces.append(patch)
    if pen_pieces:
        found["pen"] = _detection("pen", np.logical_or.reduce(pen_pieces))

    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, DARK_VALUE_MAX))
    if "mouse" in found:
        # Take the mouse's black body before the arm is built. When the mouse sat next to
        # the arm (runs/demo/20261024-230407-live/frames/230551521305.jpg) its body joined
        # the arm's dark patch, and only the blue shell was left as the mouse.
        blue = found["mouse"]
        x0, y0, x1, y1 = blue.box
        width, height = x1 - x0, y1 - y0
        zone = np.zeros(dark.shape, dtype=bool)
        zone[y0:int(y1 + MOUSE_BODY_BELOW * height),
             max(0, int(x0 - MOUSE_BODY_SIDE * width)):int(x1 + MOUSE_BODY_SIDE * width)] = True
        count, labels = cv2.connectedComponents(((dark > 0) & zone).astype(np.uint8))
        touching = _grow(blue.mask)
        body = np.zeros(dark.shape, dtype=bool)
        for label in range(1, count):
            piece = labels == label
            if (piece & touching).any():
                body |= piece
        if body.any():
            whole = body | blue.mask
            found["mouse"] = Detection("mouse", _detection("mouse", whole).box, blue.centre, whole)
            dark[body] = 0
    dark_on_table, dark_crossing = _components(dark)
    # The arm is the dark part that crosses the table edge, cut to the table.
    # The edge dips below TABLE_TOP_Y in the far right corner, so only the
    # largest piece left after the cut is the arm.
    if dark_crossing:
        below_edge = np.logical_or.reduce(dark_crossing)
        below_edge[:TABLE_TOP_Y] = False
        arm_pieces = _components(below_edge.astype(np.uint8) * 255)[0]
        if arm_pieces:
            found["arm"] = _detection("arm", arm_pieces[0])
    for patch in dark_on_table:
        owner = next((blue for blue in found.values() if (patch & _grow(blue.mask)).any()), None)
        if owner is not None:
            # The mouse body and the pen cap are dark next to their blue parts.
            # The centre stays the blue centroid, which the arm does not cover
            # when it pushes.
            whole = patch | owner.mask
            found[owner.name] = Detection(owner.name, _detection(owner.name, whole).box, owner.centre, whole)
        elif "remote" not in found:
            found["remote"] = _detection("remote", patch)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    texture = cv2.blur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (TEXTURE_BLUR, TEXTURE_BLUR))
    taken = np.zeros(frame.shape[:2], dtype=bool)
    for detection in found.values():
        taken |= _grow(detection.mask)
    # The cat toy splits the cushion's texture into pieces, so keep all of them.
    pieces = [patch for patch in _components((texture > TEXTURE_MIN).astype(np.uint8) * 255)[0] if not (patch & taken).any()]
    if pieces:
        found["cushion"] = _detection("cushion", np.logical_or.reduce(pieces))
    return found


def ends(detection: Detection) -> tuple[np.ndarray, np.ndarray]:
    """The two extreme pixels of a long object along its main axis, e.g. the pen's ends."""
    ys, xs = np.nonzero(detection.mask)
    points = np.column_stack([xs, ys]).astype(np.float64)
    centre = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - centre, full_matrices=False)
    along = (points - centre) @ axes[0]
    return centre + along.min() * axes[0], centre + along.max() * axes[0]


def draw(frame: np.ndarray, found: dict[str, Detection]) -> np.ndarray:
    out = frame.copy()
    for detection in found.values():
        x0, y0, x1, y1 = detection.box
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 255), 2)
        cv2.putText(out, detection.name, (x0, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return out
