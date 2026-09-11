import math
import random
from abc import ABC, abstractmethod
from typing import Tuple, Optional

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
    Bounces smoothly off environment boundaries to keep target visible inside canvas.
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
        raw_x = self.start_x + self.vx * t
        raw_y = self.start_y + self.vy * t

        span_x = self.max_x - 2 * self.margin
        span_y = self.max_y - 2 * self.margin

        mod_x = (raw_x - self.margin) % (2 * span_x)
        x = self.margin + (2 * span_x - mod_x if mod_x > span_x else mod_x)

        mod_y = (raw_y - self.margin) % (2 * span_y)
        y = self.margin + (2 * span_y - mod_y if mod_y > span_y else mod_y)

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

class FigureOf8Motion(BaseMotionModel):
    """
    FR-04 Mandatory Pattern 3: Figure-of-8 Motion (Lissajous Curve).
    Generates smooth figure-8 trajectory: x = cx + A*sin(w*t), y = cy + B*sin(2*w*t).
    """
    def __init__(self, center_pos: Tuple[float, float], scale_x: float = 400.0, scale_y: float = 250.0,
                 angular_speed_rad_s: float = 0.3):
        self.cx, self.cy = center_pos
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.omega = angular_speed_rad_s

    def get_position(self, t: float) -> Tuple[float, float]:
        x = self.cx + self.scale_x * math.sin(self.omega * t)
        y = self.cy + self.scale_y * math.sin(2.0 * self.omega * t)
        return float(x), float(y)

class RandomMotion(BaseMotionModel):
    """
    FR-04 Mandatory Pattern 4: Random Walk Motion.
    Generates a continuous, seed-reproducible random trajectory bounded by environment limits.
    """
    def __init__(self, start_pos: Tuple[float, float], bounds: Tuple[int, int] = (2000, 2000),
                 step_size_px: float = 50.0, random_seed: Optional[int] = 42):
        self.cx, self.cy = start_pos
        self.max_x, self.max_y = bounds
        self.step_size = step_size_px
        self.seed = random_seed
        self.margin = 50

        # Precompute deterministic trajectory points using fixed seed
        self._path = self._generate_path(steps=10000, dt_sample=0.033)

    def _generate_path(self, steps: int, dt_sample: float):
        rng = random.Random(self.seed if self.seed is not None else 42)
        x, y = self.cx, self.cy
        vx, vy = rng.uniform(-1, 1), rng.uniform(-1, 1)
        norm = math.hypot(vx, vy) or 1.0
        vx, vy = (vx / norm) * self.step_size, (vy / norm) * self.step_size

        points = [(x, y)]
        for _ in range(1, steps):
            # Smooth direction perturbation
            angle_change = rng.uniform(-0.3, 0.3)
            current_angle = math.atan2(vy, vx) + angle_change
            vx = self.step_size * math.cos(current_angle)
            vy = self.step_size * math.sin(current_angle)

            nx = x + vx * dt_sample
            ny = y + vy * dt_sample

            # Boundary bounces
            if nx < self.margin or nx > self.max_x - self.margin:
                vx = -vx
                nx = max(self.margin, min(self.max_x - self.margin, nx))
            if ny < self.margin or ny > self.max_y - self.margin:
                vy = -vy
                ny = max(self.margin, min(self.max_y - self.margin, ny))

            x, y = nx, ny
            points.append((x, y))
        return points

    def get_position(self, t: float) -> Tuple[float, float]:
        idx = int(round(t / 0.033)) % len(self._path)
        return self._path[idx]

class SpiralMotion(BaseMotionModel):
    """
    Spiral motion pattern expanding outwards from a center position.
    """
    def __init__(self, center_pos: Tuple[float, float], expansion_rate: float = 15.0, angular_speed_rad_s: float = 0.5):
        self.cx, self.cy = center_pos
        self.rate = expansion_rate
        self.omega = angular_speed_rad_s

    def get_position(self, t: float) -> Tuple[float, float]:
        r = (self.rate * t) % 400.0 + 50.0
        x = self.cx + r * math.cos(self.omega * t)
        y = self.cy + r * math.sin(self.omega * t)
        return float(x), float(y)

class SinusoidMotion(BaseMotionModel):
    """
    Sinusoidal motion pattern (wave-like trajectory).
    """
    def __init__(self, start_pos: Tuple[float, float], speed_px_s: float = 80.0, amplitude: float = 150.0, frequency: float = 0.5):
        self.start_x, self.start_y = start_pos
        self.speed = speed_px_s
        self.amp = amplitude
        self.freq = frequency

    def get_position(self, t: float) -> Tuple[float, float]:
        x = self.start_x + self.speed * t
        x = 200.0 + (x % 1600.0)
        y = self.start_y + self.amp * math.sin(2.0 * math.pi * self.freq * t)
        return float(x), float(y)

