import time
from typing import Optional, Tuple, Dict, Any
from config.schemas import SimConfig
from environment.world import VirtualEnvironment
from environment.motion import BaseMotionModel, StraightLineMotion, CircularMotion, FigureOf8Motion, RandomMotion
from target.beacon import BeaconRenderer
from camera.model import VirtualCamera
from disturbance.engine import DisturbanceEngine
from detection.base import BaseDetector
from detection.opencv_detector import OpenCVBeaconDetector
from tracking.error import TrackingErrorCalculator, TrackingErrorResult
from tracking.observation import CentroidObservation
from tracking.state import TrackingState
from tracking.state_machine import TrackingStateMachine
from reacquisition.search import ReacquisitionSearchStrategy
from pat.base import BasePATController
from pat.baseline_controller import ProportionalPATController
from metrics.engine import PerformanceMeasurementEngine, PerformanceSummaryReport
from logging_system.logger import AutomaticLogger
from frame.schema import Frame

class SimulationRunner:
    """
    FR-19 Continuous Tracking Simulator Runner (Day 2 Hardened).
    Integrates Motion Models, Beacon Rendering, Virtual Camera, Disturbance Engine,
    Detection, Tracking State Machine, Reacquisition, PAT Controller, Metrics Engine & Auto-Logger.
    """
    def __init__(self, config: Optional[SimConfig] = None,
                 motion_model: Optional[BaseMotionModel] = None,
                 detector: Optional[BaseDetector] = None,
                 controller: Optional[BasePATController] = None):
        self.config = config or SimConfig()

        # 1. Environment & Motion
        self.env = VirtualEnvironment(self.config.screen_width, self.config.screen_height)
        center = self.env.center()
        if motion_model is None:
            if self.config.motion_pattern == "circular":
                self.motion = CircularMotion(center_pos=center, radius_px=300.0, angular_speed_rad_s=0.5)
            elif self.config.motion_pattern == "figure_of_8":
                self.motion = FigureOf8Motion(center_pos=center, scale_x=400.0, scale_y=250.0)
            elif self.config.motion_pattern == "random":
                self.motion = RandomMotion(start_pos=center, bounds=(self.config.screen_width, self.config.screen_height))
            else:
                self.motion = StraightLineMotion(start_pos=center, speed_px_s=self.config.motion_speed_px_s, angle_deg=35.0)
        else:
            self.motion = motion_model

        # 2. Beacon Renderer & Virtual Camera
        self.beacon_renderer = BeaconRenderer(
            shape=self.config.target_shape,
            size_px=self.config.target_size_px,
            intensity=self.config.target_color_intensity
        )
        self.camera = VirtualCamera(self.config)

        # 3. Disturbance Engine (FR-10, FR-11, FR-12, FR-13)
        self.disturbance_engine = DisturbanceEngine()

        # 4. Detector & Reacquisition (FR-14, FR-15, FR-22)
        self.detector = detector or OpenCVBeaconDetector()
        self.reacquisition_search = ReacquisitionSearchStrategy(self.detector)

        # 5. Tracking Error Calculator & State Machine (FR-16, FR-20, FR-21)
        self.error_calc = TrackingErrorCalculator(self.config.camera_res_x, self.config.camera_res_y)
        self.state_machine = TrackingStateMachine()

        # 6. PAT Controller (FR-18)
        self.controller = controller or ProportionalPATController(self.config)

        # 7. Performance Measurement Engine & Auto-Logger (FR-23, FR-25)
        self.metrics_engine = PerformanceMeasurementEngine()
        self.logger = AutomaticLogger()

        self.sim_time = 0.0
        self._prev_state = TrackingState.SEARCHING  # Tracks previous frame state to detect state entry events

    def step(self, dt: float) -> Tuple[Frame, Optional[CentroidObservation], Optional[TrackingErrorResult], Tuple[float, float], TrackingState]:
        """
        Executes a single simulation tick of duration dt (seconds).
        Returns (degraded_frame, observation, error_result, (pan_cmd, tilt_cmd), current_state).
        """
        t0_proc = time.time()
        self.sim_time += dt

        # Step 1: Update target ground truth trajectory
        target_pos = self.motion.get_position(self.sim_time)

        # Step 2: Render clean environment canvas with beacon spot
        self.env.clear()
        self.beacon_renderer.render(self.env.canvas, target_pos)

        # Step 3: Capture clean frame crop from camera viewport
        clean_frame = self.camera.capture_frame(self.env.canvas, self.sim_time, ground_truth_world_pos=target_pos)

        # Step 4: Apply disturbance pipeline (noise + atmosphere + jitter + platform motion)
        degraded_img, dist_meta = self.disturbance_engine.process_frame(clean_frame.image, self.sim_time)

        # Package into standardized Frame object
        frame = Frame(
            image=degraded_img,
            timestamp=self.sim_time,
            frame_index=clean_frame.frame_index,
            camera_pan=clean_frame.camera_pan,
            camera_tilt=clean_frame.camera_tilt,
            ground_truth_x=clean_frame.ground_truth_x,
            ground_truth_y=clean_frame.ground_truth_y
        )

        # Step 5: Detect beacon target centroid
        if self.state_machine.state in (TrackingState.TARGET_LOST, TrackingState.REACQUISITION):
            obs = self.reacquisition_search.search(frame)
        else:
            obs = self.detector.detect(frame)

        # Step 6: Update Tracking State Machine
        current_state = self.state_machine.update(obs, self.sim_time, dt)

        # Step 7: Compute Tracking Error & PAT Control Commands
        pan_cmd, tilt_cmd = 0.0, 0.0
        error_result = None

        # ── FIX: Active Reacquisition Camera Sweep ──────────────────────────
        # ORIGINAL BUG: When target left the viewport, pan_cmd and tilt_cmd stayed
        # at 0.0 (camera frozen), so the target was never found again.
        #
        # FIX: Detect the moment we enter TARGET_LOST for the first time and
        # reset the spiral sweep origin to the current camera pointing direction.
        # Then on every REACQUISITION frame, command the camera to sweep outward
        # along a spiral search pattern until the detector finds the beacon.
        if current_state == TrackingState.TARGET_LOST and self._prev_state not in (
            TrackingState.TARGET_LOST, TrackingState.REACQUISITION
        ):
            # First frame of target loss: record current pointing as sweep center
            self.reacquisition_search.reset_sweep(
                last_known_pan=self.camera.pan_tilt.pan_deg,
                last_known_tilt=self.camera.pan_tilt.tilt_deg
            )

        if current_state in (TrackingState.TARGET_LOST, TrackingState.REACQUISITION):
            # Actively sweep camera to find the target again
            pan_cmd, tilt_cmd = self.reacquisition_search.get_sweep_command(dt)
            self.camera.update_actuator(pan_cmd, tilt_cmd, dt)
        elif obs and obs.is_valid:
            error_result = self.error_calc.compute_error(obs.centroid_x, obs.centroid_y)

            # Apply PAT control commands when tracking is active
            if current_state in (TrackingState.LOCKED, TrackingState.ACQUISITION, TrackingState.BEACON_DETECTED):
                pan_cmd, tilt_cmd = self.controller.compute_command(error_result.error_x, error_result.error_y, dt)
                self.camera.update_actuator(pan_cmd, tilt_cmd, dt)

        # Advance previous state tracker
        self._prev_state = current_state

        # Step 8: Measure processing latency and record frame telemetry
        proc_latency_ms = (time.time() - t0_proc) * 1000.0
        self.metrics_engine.record_frame(
            frame_idx=frame.frame_index,
            timestamp=self.sim_time,
            state=current_state,
            error_res=error_result,
            processing_latency_ms=proc_latency_ms,
            meta=dist_meta
        )

        return frame, obs, error_result, (pan_cmd, tilt_cmd), current_state

    def save_summary_logs(self) -> Tuple[str, str]:
        summary_report = self.metrics_engine.generate_summary(
            acquisition_time_sec=self.state_machine.acquisition_time_sec,
            reacquisition_time_sec=self.state_machine.last_reacquisition_time_sec
        )
        return self.logger.save_run_logs(self.metrics_engine.frame_records, summary_report)
