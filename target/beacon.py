import cv2
import numpy as np
from typing import Tuple

class BeaconRenderer:
    """
    FR-03 Moving Beacon Generation.
    Renders a bright optical beacon spot (default square, 10x10 px) into the
    environment canvas at the target's current ground-truth position (x, y).
    """
    def __init__(self, shape: str = "square", size_px: int = 10, intensity: int = 255):
        self.shape = shape
        self.size_px = max(5, min(20, size_px))  # Clamped to PS range 5-20 px
        self.intensity = intensity

    def render(self, canvas: np.ndarray, position: Tuple[float, float]) -> np.ndarray:
        """
        Renders the beacon spot onto the provided canvas in-place (or returns updated canvas).
        """
        x, y = position
        half_size = self.size_px / 2.0

        if self.shape == "square":
            x1 = int(round(x - half_size))
            y1 = int(round(y - half_size))
            x2 = int(round(x + half_size))
            y2 = int(round(y + half_size))

            # Clip bounding rectangle to environment canvas bounds
            h, w = canvas.shape[:2]
            x1_clamped = max(0, min(w, x1))
            y1_clamped = max(0, min(h, y1))
            x2_clamped = max(0, min(w, x2))
            y2_clamped = max(0, min(h, y2))

            if x2_clamped > x1_clamped and y2_clamped > y1_clamped:
                canvas[y1_clamped:y2_clamped, x1_clamped:x2_clamped] = self.intensity

        elif self.shape == "circle":
            center_int = (int(round(x)), int(round(y)))
            radius = int(round(half_size))
            cv2.circle(canvas, center_int, radius, color=self.intensity, thickness=-1)

        return canvas
