"""The Logitech C270 on the desk partition, looking back at the arm."""

import threading
import time
from pathlib import Path

import cv2
import numpy as np

# The by-id path survives the USB hub re-plugging, /dev/videoN does not. Set it to your
# camera's entry in /dev/v4l/by-id/.
DEVICE = "/dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_XXXXXXXX-video-index0"
# The first frames are black while auto exposure settles.
WARMUP_FRAMES = 40
# A frame must start this long after snap() is called. The driver can hand out a frame
# that was exposed up to a frame period before it is read, about 33 ms at 30 fps.
FRESH_AFTER_S = 0.05
SNAP_TIMEOUT_S = 2.0


class Camera:
    """The webcam read continuously on a thread, so snap() returns a frame taken after the call.

    Dropping a fixed number of buffered frames was not enough: on 24 Oct, with the camera
    left unread for about 20 s during a push, snap() returned frames from the middle of
    the push (runs/demo/20261024-225435, frames 5 and 8), and the push measurements and
    the "touching" tick were built on them.
    """

    def __init__(self, device: str = DEVICE) -> None:
        self.device = device
        self.capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open {device}")
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(WARMUP_FRAMES):
            ok, _ = self.capture.read()
            if not ok:
                raise RuntimeError(f"no frame from {device}")
        self.latest: tuple[float, np.ndarray] | None = None
        self.error: str | None = None
        self.condition = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while self.running:
            ok, frame = self.capture.read()
            stamp = time.monotonic()
            with self.condition:
                if not ok:
                    self.error = f"no frame from {self.device}"
                else:
                    self.latest = (stamp, frame)
                self.condition.notify_all()
            if not ok:
                return

    def snap(self) -> np.ndarray:
        """The first frame read at least FRESH_AFTER_S after this call."""
        wanted = time.monotonic() + FRESH_AFTER_S
        deadline = wanted + SNAP_TIMEOUT_S
        with self.condition:
            while True:
                if self.error is not None:
                    raise RuntimeError(self.error)
                if self.latest is not None and self.latest[0] >= wanted:
                    return self.latest[1].copy()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"no fresh frame from {self.device} in {SNAP_TIMEOUT_S} s")
                self.condition.wait(remaining)

    def close(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        self.capture.release()


def snap(device: str = DEVICE) -> np.ndarray:
    camera = Camera(device)
    try:
        return camera.snap()
    finally:
        camera.close()


def save(frame: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"could not write {path}")
