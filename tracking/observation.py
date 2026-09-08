from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class CentroidObservation:
    """
    FR-14 & FR-15 Detection & Centroiding Observation Output.
    Represents detected target centroid in camera frame pixel space.
    """
    centroid_x: float          # Sub-pixel X coordinate in camera frame (0..640)
    centroid_y: float          # Sub-pixel Y coordinate in camera frame (0..480)
    confidence: float          # Detection confidence score (0.0 to 1.0)
    bbox: Tuple[int, int, int, int]  # Bounding box (x, y, width, height) in frame
    area_px: float             # Area of detected blob in pixels
    is_valid: bool = True      # Flag indicating if a valid beacon was detected
