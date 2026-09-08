import math
from abc import ABC, abstractmethod
from typing import Tuple

class BaseMotionModel(ABC):
    """Abstract Base Class for Target Motion Trajectory Generators (FR-04)."""
    @abstractmethod
    def get_position(self, t: float) -> Tuple[float, float]:
        """Returns target ground-truth (x, y) coordinates in world space at time t (sec)."""
        pass

class StraightLineMotion(BaseMotionModel):
    """
    FR-04 Mandatory Pattern 1: Straight Line Motion.
    Moves in a straight line at configured speed (px/s) and angle (rad/deg).
    Bounces smoothly off environment boundaries to keep target visible inside 2000x2000 canvas.
    """
    def __init__(self, start_pos: Tuple[float, float], speed_px_s: float = 80.0, angle_deg: float = 35.0,
                 bounds: Tuple[int, int] = (2000, 2000), margin: int = 20):
        self.start_x, self.start_y = start_pos
        self.speed = speed_px_s
        self.angle_rad = math.radians(angle_deg)
        self.vx = self.speed * math.cos(self.angle_rad)
        self.vy = self.speed * math.sin(self.angle_rad)
        self.max_x, self.max_y = bounds
        self.margin = margin

    def get_position(self, t: float) -> Tuple[float, float]:
        # Unbounded linear trajectory
        raw_x = self.start_x + self.vx * t
        raw_y = self.start_y + self.vy * t

        # Smooth triangular wave folding to stay inside canvas bounds
        span_x = self.max_x - 2 * self.margin
        span_y = self.max_y - 2 * self.margin

        mod_x = (raw_x - self.margin) % (2 * span_x)
        if mod_x > span_x:
            x = self.margin + 2 * span_x - mod_x
        else:
            x = self.margin + mod_x

        mod_y = (raw_y - self.margin) % (2 * span_y)
        if mod_y > span_y:
            y = self.margin + 2 * span_y - mod_y
        else:
            y = self.margin + mod_y

        return float(x), float(y)

class CircularMotion(BaseMotionModel):
    """
    FR-04 Mandatory Pattern 2: Circular Motion.
    Orbits around a center position with radius R (px) and angular speed (rad/s).
    """
    def __init__(self, center_pos: Tuple[float, float], radius_px: float = 350.0,
                 angular_speed_rad_s: float = 0.4):
        self.cx, self.cy = center_pos
        self.radius = radius_px
        self.omega = angular_speed_rad_s

    def get_position(self, t: float) -> Tuple[float, float]:
        x = self.cx + self.radius * math.cos(self.omega * t)
        y = self.cy + self.radius * math.sin(self.omega * t)
        return float(x), float(y)
