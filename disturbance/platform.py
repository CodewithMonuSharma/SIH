import cv2
import numpy as np
import math
from typing import Tuple

class PlatformMotionEngine:
    """
    FR-12 Platform Motion Subsystem.
    Simulates correlated host platform movement drift (Linear default, max +-20 px/frame).
    """
    def __init__(self, pattern: str = "Linear", max_drift_px: float = 10.0, speed_hz: float = 0.2):
        self.pattern = pattern
        self.max_drift = min(20.0, max(0.0, max_drift_px))
        self.speed_hz = speed_hz

    def get_offset(self, t: float) -> Tuple[float, float]:
        """Calculates correlated platform drift offset (dx, dy) at time t."""
        if self.max_drift <= 0:
            return 0.0, 0.0

        if self.pattern == "Linear":
            # Smooth low-frequency sinusoidal drift
            dx = self.max_drift * math.sin(2.0 * math.pi * self.speed_hz * t)
            dy = self.max_drift * math.cos(2.0 * math.pi * self.speed_hz * t)
            return dx, dy
        elif self.pattern == "Circular":
            dx = self.max_drift * math.cos(2.0 * math.pi * self.speed_hz * t)
            dy = self.max_drift * math.sin(2.0 * math.pi * self.speed_hz * t)
            return dx, dy

        return 0.0, 0.0

    def apply_platform_motion(self, image: np.ndarray, t: float) -> Tuple[np.ndarray, Tuple[float, float]]:
        dx, dy = self.get_offset(t)
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return image, (0.0, 0.0)

        M = np.float32([[1, 0, dx], [0, 1, dy]])
        h, w = image.shape[:2]
        shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return shifted, (dx, dy)
