import time
from typing import Optional, Tuple, Dict, Any
from config.schemas import SimConfig
from environment.world import VirtualEnvironment
from environment.motion import BaseMotionModel, StraightLineMotion, CircularMotion
from target.beacon import BeaconRenderer
from camera.model import VirtualCamera
from detection.base import BaseDetector
from detection.opencv_detector import OpenCVBeaconDetector
from tracking.error import TrackingErrorCalculator, TrackingErrorResult
from tracking.observation import CentroidObservation
from pat.base import BasePATController
from pat.baseline_controller import ProportionalPATController
from frame.schema import Frame

class SimulationRunner:
    """
    FR-19 Continuous Tracking Simulator Runner.
    Wires the closed loop:
    Target Motion -> Beacon Render -> Virtual Camera Frame -> Detection ->
    Centroid Estimation -> Tracking Error -> PAT Controller -> Actuator Repositioning.
    """
    def __init__(self, config: Optional[SimConfig] = None,
                 motion_model: Optional[BaseMotionModel] = None,
                 detector: Optional[BaseDetector] = None,
                 controller: Optional[BasePATController] = None):
        self.config = config or SimConfig()

        # Initialize Environment
        self.env = VirtualEnvironment(self.config.screen_width, self.config.screen_height)

        # Initialize Motion Model
        if motion_model is None:
            center = self.env.center()
            if self.config.motion_pattern == "circular":
                self.motion = CircularMotion(center_pos=center, radius_px=300.0, angular_speed_rad_s=0.5)
            else:
                self.motion = StraightLineMotion(start_pos=center, speed_px_s=self.config.motion_speed_px_s, angle_deg=35.0)
        else:
            self.motion = motion_model

        # Initialize Target Beacon Renderer
        self.beacon_renderer = BeaconRenderer(
            shape=self.config.target_shape,
            size_px=self.config.target_size_px,
            intensity=self.config.target_color_intensity
        )

        # Initialize Camera
        self.camera = VirtualCamera(self.config)

        # Initialize Detector (swappable)
        self.detector = detector or OpenCVBeaconDetector()

        # Initialize Tracking Error Calculator
        self.error_calc = TrackingErrorCalculator(self.config.camera_res_x, self.config.camera_res_y)

        # Initialize PAT Controller (swappable)
        self.controller = controller or ProportionalPATController(self.config)

        # Time & Step tracking
        self.sim_time = 0.0

    def step(self, dt: float) -> Tuple[Frame, Optional[CentroidObservation], Optional[TrackingErrorResult], Tuple[float, float]]:
        """
        Executes a single simulation tick of duration dt (seconds).
        Returns (frame, observation, error_result, (pan_cmd, tilt_cmd)).
        """
        self.sim_time += dt

        # Step 1: Update target motion ground truth
        target_pos = self.motion.get_position(self.sim_time)

        # Step 2: Render environment canvas with beacon spot
        self.env.clear()
        self.beacon_renderer.render(self.env.canvas, target_pos)

        # Step 3: Capture camera frame
        frame = self.camera.capture_frame(self.env.canvas, self.sim_time, ground_truth_world_pos=target_pos)

        # Step 4: Detect target & compute centroid
        obs = self.detector.detect(frame)

        # Step 5: Compute tracking error & PAT control commands
        pan_cmd, tilt_cmd = 0.0, 0.0
        error_result = None

        if obs and obs.is_valid:
            error_result = self.error_calc.compute_error(obs.centroid_x, obs.centroid_y)
            pan_cmd, tilt_cmd = self.controller.compute_command(error_result.error_x, error_result.error_y, dt)

            # Step 6: Apply PAT control command to camera pan-tilt actuator
            self.camera.update_actuator(pan_cmd, tilt_cmd, dt)

        return frame, obs, error_result, (pan_cmd, tilt_cmd)
