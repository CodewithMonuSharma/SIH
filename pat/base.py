from abc import ABC, abstractmethod
from typing import Tuple

class BasePATController(ABC):
    """
    FR-18 PAT Controller Interface & Section 7 Architectural Backbone.
    Converts centroid tracking error (error_x, error_y in pixels) into
    pan/tilt angular velocity commands (deg/sec).
    """
    @abstractmethod
    def compute_command(self, error_x: float, error_y: float, dt: float) -> Tuple[float, float]:
        """
        Returns (pan_cmd_deg_s, tilt_cmd_deg_s).
        """
        pass
