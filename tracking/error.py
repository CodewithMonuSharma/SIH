import math
from dataclasses import dataclass
from typing import Tuple, Optional, List

@dataclass
class TrackingErrorResult:
    error_x: float           # Pixel offset in X from frame boresight (center)
    error_y: float           # Pixel offset in Y from frame boresight (center)
    error_magnitude: float   # Euclidean pixel distance to frame center
    running_rmse: float      # Accumulated Root-Mean-Square Error across run

class TrackingErrorCalculator:
    """
    FR-16 Tracking Error Calculation.
    Computes per-frame Euclidean pixel distance between detected target centroid
    and camera boresight center (320, 240 for 640x480 frame), maintaining running RMSE.
    """
    def __init__(self, frame_res_x: int = 640, frame_res_y: int = 480):
        self.boresight_x = frame_res_x / 2.0
        self.boresight_y = frame_res_y / 2.0
        self.squared_errors: List[float] = []

    def compute_error(self, centroid_x: float, centroid_y: float) -> TrackingErrorResult:
        error_x = centroid_x - self.boresight_x
        error_y = centroid_y - self.boresight_y
        error_mag = math.sqrt(error_x**2 + error_y**2)

        self.squared_errors.append(error_mag**2)
        mean_sq_error = sum(self.squared_errors) / len(self.squared_errors)
        running_rmse = math.sqrt(mean_sq_error)

        return TrackingErrorResult(
            error_x=error_x,
            error_y=error_y,
            error_magnitude=error_mag,
            running_rmse=running_rmse
        )

    def reset(self):
        self.squared_errors.clear()
