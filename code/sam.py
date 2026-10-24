"""Object masks from scripts/sam_server.py (OWLv2 boxes, SAM 2.1 masks) on a remote GPU.

The server is reached through an ssh tunnel to 127.0.0.1:23101. Only the camera frame is
sent, as a JPEG. detect() returns the
same Detection objects as cap_so101/detect.py, so the desk and the skills use either.
Failures raise: a missing server is never replaced by a guess.
"""

import base64
import json
import urllib.request

import cv2
import numpy as np

from cap_so101.detect import Detection, _detection

URL = "http://127.0.0.1:23101"
# One prompt per object on the desk. The arm is asked for too, so it can be told apart from
# what it touches.
PROMPTS = {
    "mouse": "a computer mouse",
    "pen": "a pen",
    "remote": "a TV remote control",
    "cushion": "a grey fluffy cushion",
    "arm": "a black robot arm",
}
# Boxes larger than this share of the frame are not one desk object.
MAX_BOX_FRACTION = 0.3
TIMEOUT_S = 15.0


def segment(frame: np.ndarray, prompts: dict[str, str] = PROMPTS) -> dict:
    ok, jpeg = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("could not encode the frame")
    body = json.dumps({"image_jpeg_base64": base64.b64encode(jpeg.tobytes()).decode(),
                       "prompts": prompts, "max_box_fraction": MAX_BOX_FRACTION}).encode()
    request = urllib.request.Request(f"{URL}/segment", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.loads(response.read())


def detect(frame: np.ndarray, prompts: dict[str, str] = PROMPTS) -> dict[str, Detection]:
    """Objects by name from SAM masks. Where two masks overlap, the pixel goes to the higher score."""
    reply = segment(frame, prompts)
    masks = {}
    for name, found in reply["objects"].items():
        png = np.frombuffer(base64.b64decode(found["mask_png_base64"]), np.uint8)
        mask = cv2.imdecode(png, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"could not decode the mask for {name}")
        masks[name] = (found["score"], mask > 0)
    found_objects: dict[str, Detection] = {}
    for name, (score, mask) in masks.items():
        for other, (other_score, other_mask) in masks.items():
            if other != name and other_score > score:
                mask = mask & ~other_mask
        if mask.any():
            found_objects[name] = _detection(name, mask)
    return found_objects
