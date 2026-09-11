from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class SimConfig:
    """
    Central Configuration Store for PS 26169 (CoarsePAT-Sim).
    Holds all parameters with exact official PS-26169 default values.
    """
    # Virtual Environment Canvas
    screen_width: int = 2000
    screen_height: int = 2000
    
    # Camera Parameters
    camera_res_x: int = 640
    camera_res_y: int = 480
    camera_fov_h_deg: float = 4.0
    camera_fov_v_deg: float = 3.0
    camera_update_rate_hz: float = 30.0
    
    # Actuator / Pan-Tilt Controls
    max_pan_speed_deg_s: float = 5.0
    max_tilt_speed_deg_s: float = 5.0
    control_update_rate_hz: float = 30.0
    
    # Beacon / Target Parameters
    target_type: str = "Beacon Spot"
    target_shape: str = "circle"
    target_size_px: int = 10
    target_color_intensity: int = 255
    
    # Motion Parameters
    motion_pattern: str = "straight_line"  # "straight_line", "circular", "figure_of_8", "random"
    motion_speed_px_s: float = 40.0
    
    # Mapping Scale: Pixels per degree of FOV
    # e.g., 640 px corresponds to 4.0 degrees FOV => 160 px/degree
    @property
    def px_per_degree_h(self) -> float:
        return self.camera_res_x / self.camera_fov_h_deg

    @property
    def px_per_degree_v(self) -> float:
        return self.camera_res_y / self.camera_fov_v_deg

    # Canvas FOV in world pixels
    @property
    def fov_width_px(self) -> float:
        return self.camera_res_x

    @property
    def fov_height_px(self) -> float:
        return self.camera_res_y
