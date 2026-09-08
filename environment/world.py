import numpy as np
from typing import Tuple

class VirtualEnvironment:
    """
    FR-02 Virtual Environment: 2D World Canvas space (≥2000x2000 px).
    Maintains the full coordinate space where the target beacon moves
    and from which the virtual camera crops its FOV sub-window.
    """
    def __init__(self, width: int = 2000, height: int = 2000):
        if width < 2000 or height < 2000:
            raise ValueError(f"Environment dimensions must be >= 2000x2000 px, got {width}x{height}")
        self.width = width
        self.height = height
        # Internal canvas memory (uint8 monochrome, 0 = dark space background)
        self._canvas = np.zeros((self.height, self.width), dtype=np.uint8)

    def clear(self):
        """Reset environment canvas to background (dark)."""
        self._canvas.fill(0)

    @property
    def canvas(self) -> np.ndarray:
        return self._canvas

    def center(self) -> Tuple[float, float]:
        """Returns the world center coordinates (x, y)."""
        return self.width / 2.0, self.height / 2.0
