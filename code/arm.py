"""The SO-101 arm on its Feetech bus, in calibrated joint units.

Joints 1 to 5 are in degrees from the middle of their calibrated range and the
gripper is in percent of its range, the same formulas as LeRobot 0.6.1's
DEGREES and RANGE_0_100 modes. scripts/calibrate.py wrote the homing offsets
into the motors and the ranges into so101/lerobot/cap_so101.json, so the bus
reads raw steps and this file does the conversion. The arm refuses to move
until so101/joint_limits.json exists, which scripts/read_joints.py
--record-limits writes while the arm is moved by hand.
"""

import json
import time
from pathlib import Path

import numpy as np

from cap_so101.feetech import FeetechMotorsBus

REPO = Path(__file__).resolve().parent.parent
# The by-id path survives the USB hub re-plugging, /dev/ttyACM0 does not. Set it to your
# bus adapter's entry in /dev/serial/by-id/.
PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_XXXXXXXXXX-if00"
CALIBRATION = REPO / "so101" / "lerobot" / "cap_so101.json"
LIMITS = REPO / "so101" / "joint_limits.json"

MOTORS = {
    "shoulder_pan": (1, "sts3215"),
    "shoulder_lift": (2, "sts3215"),
    "elbow_flex": (3, "sts3215"),
    "wrist_flex": (4, "sts3215"),
    "wrist_roll": (5, "sts3215"),
    "gripper": (6, "sts3215"),
}
JOINT_NAMES = list(MOTORS)

# Slow on purpose: the first runs are next to a person and a plush cat.
MAX_SPEED_PER_S = 30.0
STEP_S = 0.02
STEPS_PER_TURN = 4095
# Two reads this far apart that differ by less than the tolerance count as stopped.
SETTLE_INTERVAL_S = 0.1
SETTLE_TOLERANCE = 0.2
SETTLE_TIMEOUT_S = 2.0


class Arm:
    def __init__(self, port: str = PORT, calibration_path: Path = CALIBRATION) -> None:
        calibration = json.loads(calibration_path.read_text())
        self.range_min = np.array([calibration[name]["range_min"] for name in JOINT_NAMES], dtype=np.float64)
        self.range_max = np.array([calibration[name]["range_max"] for name in JOINT_NAMES], dtype=np.float64)
        self.bus = FeetechMotorsBus(port=port, motors=MOTORS)
        self.bus.connect()
        self.limits = json.loads(LIMITS.read_text()) if LIMITS.exists() else None

    def to_calibrated(self, raw: np.ndarray) -> np.ndarray:
        middle = (self.range_min + self.range_max) / 2
        calibrated = (raw - middle) * 360 / STEPS_PER_TURN
        calibrated[-1] = (raw[-1] - self.range_min[-1]) / (self.range_max[-1] - self.range_min[-1]) * 100
        return calibrated

    def to_raw(self, calibrated: np.ndarray) -> np.ndarray:
        middle = (self.range_min + self.range_max) / 2
        raw = calibrated * STEPS_PER_TURN / 360 + middle
        raw[-1] = calibrated[-1] / 100 * (self.range_max[-1] - self.range_min[-1]) + self.range_min[-1]
        return np.round(raw).astype(np.int32)

    def read_raw(self) -> np.ndarray:
        return np.asarray(self.bus.read("Present_Position"), dtype=np.float64)

    def read(self) -> np.ndarray:
        return self.to_calibrated(self.read_raw())

    def set_torque(self, enabled: bool) -> None:
        if enabled:
            # On 24 Oct every Goal_Position read 0 after power-up, and torque on
            # drives each joint to its goal, so hold the present pose first.
            self.bus.write("Goal_Position", self.read_raw().astype(np.int32))
        self.bus.write("Torque_Enable", 1 if enabled else 0)

    def move(self, target: np.ndarray) -> np.ndarray:
        """Go to target in a straight line in joint space, then return the joints read back."""
        return self.follow([target])

    def follow(self, waypoints: list[np.ndarray]) -> np.ndarray:
        """Go through waypoints in straight joint-space lines at the speed cap, settling once at the end.

        Settling after every 2 cm push segment stopped the mouse about ten times per push.
        Each segment starts from the previous goal, not a read, since the motors lag the goal.
        """
        if self.limits is None:
            raise RuntimeError(f"{LIMITS} is missing. Record limits by hand first.")
        lower = np.array([self.limits[name][0] for name in JOINT_NAMES])
        upper = np.array([self.limits[name][1] for name in JOINT_NAMES])
        start = self.read()
        for target in waypoints:
            clamped = np.clip(np.asarray(target, dtype=np.float64), lower, upper)
            steps = max(1, int(np.ceil(np.abs(clamped - start).max() / MAX_SPEED_PER_S / STEP_S)))
            for step in range(1, steps + 1):
                self.bus.write("Goal_Position", self.to_raw(start + (clamped - start) * step / steps))
                time.sleep(STEP_S)
            start = clamped
        return self.settle()

    def settle(self) -> np.ndarray:
        """Read the joints once they stop moving, or after SETTLE_TIMEOUT_S if they never do.

        The motors lag the last goal. On 24 Oct shoulder_pan read 2.3 degrees short
        right after a 5 degree move and 0.3 short a few seconds later. After the
        timeout a large error is a stall, not lag.
        """
        previous = self.read()
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while time.monotonic() < deadline:
            time.sleep(SETTLE_INTERVAL_S)
            current = self.read()
            if np.abs(current - previous).max() < SETTLE_TOLERANCE:
                return current
            previous = current
        return previous

    def close(self) -> None:
        self.bus.disconnect()
