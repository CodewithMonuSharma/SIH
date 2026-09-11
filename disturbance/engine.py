import numpy as np
from typing import Tuple, Dict, Any
from disturbance.noise import ImageNoiseEngine
from disturbance.atmosphere import AtmosphericConditionEngine
from disturbance.jitter import CameraJitterEngine
from disturbance.platform import PlatformMotionEngine

class DisturbanceEngine:
    """
    FR-10, FR-11, FR-12, FR-13 Master Disturbance Subsystem.
    Composites Image Noise, Atmospheric Degradation, Camera Jitter, and Platform Motion
    onto clean camera frames before passing to detection.
    """
    def __init__(self, random_seed: int = 42):
        self.noise_engine = ImageNoiseEngine(random_seed=random_seed)
        self.jitter_engine = CameraJitterEngine(max_jitter_px=0.0, seed=random_seed)
        self.platform_engine = PlatformMotionEngine(pattern="Linear", max_drift_px=0.0)

        # Disturbance configuration settings
        self.enable_gaussian = False
        self.gaussian_std_dev = 15.0

        self.enable_salt_pepper = False
        self.sp_density = 0.05

        self.enable_poisson = False
        self.poisson_scale = 1.0

        self.atmospheric_condition = "Clear"
        self.atmosphere_severity = 0.5

        self.enable_jitter = False
        self.jitter_max_px = 10.0

        self.enable_platform_motion = False
        self.platform_max_px = 10.0

    def process_frame(self, frame_image: np.ndarray, timestamp: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies all configured disturbances in correct physical/sensor order:
        Clean Frame -> Atmospheric Effect -> Image/Sensor Noise -> Camera Jitter / Platform Shift.
        """
        img = frame_image.copy()
        meta = {}

        # Step 1: Atmospheric Degradation (FR-11)
        if self.atmospheric_condition != "Clear":
            img = AtmosphericConditionEngine.apply_condition(
                img, self.atmospheric_condition, self.atmosphere_severity
            )
            meta["atmosphere"] = self.atmospheric_condition

        # Step 2: Sensor / Image Noise Injection (FR-10)
        if self.enable_gaussian:
            img = self.noise_engine.apply_gaussian_noise(img, self.gaussian_std_dev)
            meta["gaussian_std"] = self.gaussian_std_dev

        if self.enable_salt_pepper:
            img = self.noise_engine.apply_salt_and_pepper_noise(img, self.sp_density)
            meta["sp_density"] = self.sp_density

        if self.enable_poisson:
            img = self.noise_engine.apply_poisson_noise(img, self.poisson_scale)
            meta["poisson_scale"] = self.poisson_scale

        # Step 3: Platform Motion Drift (FR-12)
        if self.enable_platform_motion and self.platform_max_px > 0:
            self.platform_engine.max_drift = self.platform_max_px
            img, (p_dx, p_dy) = self.platform_engine.apply_platform_motion(img, timestamp)
            meta["platform_offset"] = (p_dx, p_dy)

        # Step 4: Camera Frame Jitter (FR-13)
        if self.enable_jitter and self.jitter_max_px > 0:
            self.jitter_engine.max_jitter = self.jitter_max_px
            img, (j_dx, j_dy) = self.jitter_engine.apply_jitter(img)
            meta["jitter_offset"] = (j_dx, j_dy)

        return img, meta
