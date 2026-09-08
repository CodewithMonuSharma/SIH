from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Frame:
    """
    Standardized Frame Interface Contract (FR-09).
    This single interface boundary is consumed identically by Detection, Centroiding,
    Tracking, and PAT Controller modules regardless of whether the source is a
    Virtual Camera or an Ingested MP4.
    """
    image: np.ndarray             # 2D/3D uint8 numpy array (monochrome or BGR)
    timestamp: float              # Simulation or video timestamp in seconds
    frame_index: int              # Sequential frame counter (0-indexed)
    camera_pan: float = 0.0       # Current camera pointing pan angle (deg)
    camera_tilt: float = 0.0      # Current camera pointing tilt angle (deg)
    ground_truth_x: Optional[float] = None  # World coordinate X (sim mode ground truth)
    ground_truth_y: Optional[float] = None  # World coordinate Y (sim mode ground truth)

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]
