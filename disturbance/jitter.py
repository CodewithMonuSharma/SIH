import cv2
import numpy as np
from typing import Tuple

class CameraJitterEngine:
    """
    FR-13 Camera Jitter Subsystem.
    Applies small, uncorrelated frame-to-frame pixel shifts up to configured
    maximum magnitude (default max +-20 px/frame).
    """
    def __init__(self, max_jitter_px: float = 10.0, seed: int = 42):
        self.max_jitter = min(20.0, max(0.0, max_jitter_px))
        self.rng = np.random.default_rng(seed)

    def apply_jitter(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Applies an affine shift matrix to the image and returns (jittered_image, (dx, dy)).
        """
        if self.max_jitter <= 0:
            return image, (0.0, 0.0)

        dx = float(self.rng.uniform(-self.max_jitter, self.max_jitter))
        dy = float(self.rng.uniform(-self.max_jitter, self.max_jitter))

        M = np.float32([[1, 0, dx], [0, 1, dy]])
        h, w = image.shape[:2]
        jittered = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return jittered, (dx, dy)
