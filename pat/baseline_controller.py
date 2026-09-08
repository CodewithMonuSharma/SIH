from typing import Tuple
from pat.base import BasePATController
from config.schemas import SimConfig

class ProportionalPATController(BasePATController):
    """
    FR-18 Baseline PAT Controller (Proportional P-Controller).
    Converts pixel tracking error into proportional angular velocity commands
    bounded by maximum pan/tilt speed constraints (default 5.0 °/s).
    """
    def __init__(self, config: SimConfig, kp: float = 8.0):
        self.config = config
        self.kp = kp
        self.max_pan_speed = config.max_pan_speed_deg_s
        self.max_tilt_speed = config.max_tilt_speed_deg_s
        self.px_per_deg_h = config.px_per_degree_h
        self.px_per_deg_v = config.px_per_degree_v

    def compute_command(self, error_x: float, error_y: float, dt: float) -> Tuple[float, float]:
        # Convert pixel error to angular error (deg)
        deg_error_x = error_x / self.px_per_deg_h
        deg_error_y = error_y / self.px_per_deg_v

        # Proportional control law
        raw_pan_cmd = self.kp * deg_error_x
        raw_tilt_cmd = self.kp * deg_error_y

        # Velocity clamping (FR-08)
        pan_cmd = max(-self.max_pan_speed, min(self.max_pan_speed, raw_pan_cmd))
        tilt_cmd = max(-self.max_tilt_speed, min(self.max_tilt_speed, raw_tilt_cmd))

        return pan_cmd, tilt_cmd
