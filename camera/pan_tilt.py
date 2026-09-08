from typing import Tuple

class PanTiltMechanism:
    """
    FR-07 Pan/Tilt Simulation & FR-08 Camera Motion Constraints.
    Applies commanded pan/tilt velocity to the camera's pointing direction,
    clamped to configured maximum speed (5-10°/s, default 5°/s).
    """
    def __init__(self, initial_pan_deg: float = 0.0, initial_tilt_deg: float = 0.0,
                 max_pan_speed_deg_s: float = 5.0, max_tilt_speed_deg_s: float = 5.0):
        self.pan_deg = initial_pan_deg
        self.tilt_deg = initial_tilt_deg
        self.max_pan_speed = max_pan_speed_deg_s
        self.max_tilt_speed = max_tilt_speed_deg_s
        self.last_pan_cmd = 0.0
        self.last_tilt_cmd = 0.0

    def update(self, pan_cmd_deg_s: float, tilt_cmd_deg_s: float, dt: float) -> Tuple[float, float]:
        """
        Updates pointing angles based on pan/tilt velocity commands over time step dt.
        Enforces speed caps (FR-08).
        """
        # Clamp velocity commands
        clamped_pan_cmd = max(-self.max_pan_speed, min(self.max_pan_speed, pan_cmd_deg_s))
        clamped_tilt_cmd = max(-self.max_tilt_speed, min(self.max_tilt_speed, tilt_cmd_deg_s))

        self.last_pan_cmd = clamped_pan_cmd
        self.last_tilt_cmd = clamped_tilt_cmd

        # Integrate angles
        self.pan_deg += clamped_pan_cmd * dt
        self.tilt_deg += clamped_tilt_cmd * dt

        return self.pan_deg, self.tilt_deg

    def set_pointing(self, pan_deg: float, tilt_deg: float):
        """Directly set camera pointing angle (e.g. for reset/initialization)."""
        self.pan_deg = pan_deg
        self.tilt_deg = tilt_deg
