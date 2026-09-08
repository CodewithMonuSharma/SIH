import pytest
import numpy as np
from config.schemas import SimConfig
from environment.world import VirtualEnvironment
from environment.motion import StraightLineMotion, CircularMotion
from target.beacon import BeaconRenderer
from camera.model import VirtualCamera
from camera.viewport import CameraViewport
from camera.pan_tilt import PanTiltMechanism
from detection.opencv_detector import OpenCVBeaconDetector
from tracking.error import TrackingErrorCalculator
from pat.baseline_controller import ProportionalPATController
from runner.simulator import SimulationRunner

def test_sim_config_defaults():
    config = SimConfig()
    assert config.screen_width == 2000
    assert config.screen_height == 2000
    assert config.camera_res_x == 640
    assert config.camera_res_y == 480
    assert config.camera_fov_h_deg == 4.0
    assert config.camera_fov_v_deg == 3.0
    assert config.max_pan_speed_deg_s == 5.0
    assert config.max_tilt_speed_deg_s == 5.0
    assert config.target_size_px == 10
    assert config.px_per_degree_h == 160.0
    assert config.px_per_degree_v == 160.0

def test_virtual_environment():
    env = VirtualEnvironment(2000, 2000)
    assert env.canvas.shape == (2000, 2000)
    assert env.center() == (1000.0, 1000.0)
    
    with pytest.raises(ValueError):
        VirtualEnvironment(1000, 1000)

def test_motion_models():
    motion_line = StraightLineMotion(start_pos=(1000, 1000), speed_px_s=50.0, angle_deg=0.0)
    x0, y0 = motion_line.get_position(0.0)
    assert x0 == 1000.0 and y0 == 1000.0
    x1, y1 = motion_line.get_position(1.0)
    assert x1 == 1050.0 and y1 == 1000.0

    motion_circle = CircularMotion(center_pos=(1000, 1000), radius_px=200.0, angular_speed_rad_s=0.5)
    cx0, cy0 = motion_circle.get_position(0.0)
    assert pytest.approx(cx0, 0.1) == 1200.0
    assert pytest.approx(cy0, 0.1) == 1000.0

def test_beacon_renderer_and_detection():
    env = VirtualEnvironment(2000, 2000)
    renderer = BeaconRenderer(shape="square", size_px=10, intensity=255)
    target_pos = (1000.0, 1000.0)
    renderer.render(env.canvas, target_pos)

    config = SimConfig()
    camera = VirtualCamera(config)
    frame = camera.capture_frame(env.canvas, timestamp=0.0, ground_truth_world_pos=target_pos)

    assert frame.image.shape == (480, 640)
    assert frame.ground_truth_x == 1000.0
    assert frame.ground_truth_y == 1000.0

    detector = OpenCVBeaconDetector(threshold_value=200)
    obs = detector.detect(frame)
    assert obs is not None
    assert obs.is_valid
    # Boresight center of camera frame is 320, 240
    assert pytest.approx(obs.centroid_x, abs=1.0) == 320.0
    assert pytest.approx(obs.centroid_y, abs=1.0) == 240.0

def test_pat_controller_and_clamping():
    config = SimConfig(max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=5.0)
    controller = ProportionalPATController(config, kp=1.5)

    # Moderate tracking error
    pan_cmd, tilt_cmd = controller.compute_command(error_x=16.0, error_y=-16.0, dt=0.033)
    # 16 px / 160 px/deg = 0.1 deg => 1.5 * 0.1 = 0.15 deg/s
    assert pytest.approx(pan_cmd, abs=0.01) == 0.15
    assert pytest.approx(tilt_cmd, abs=0.01) == -0.15

    # Large error should clamp to max speed 5.0 deg/s
    pan_cmd_max, tilt_cmd_max = controller.compute_command(error_x=1000.0, error_y=-1000.0, dt=0.033)
    assert pan_cmd_max == 5.0
    assert tilt_cmd_max == -5.0

def test_closed_loop_simulator_convergence():
    config = SimConfig()
    runner = SimulationRunner(config)

    # Run closed loop for 100 frames (~3.3 seconds)
    dt = 1.0 / 30.0
    final_error = None
    for _ in range(100):
        frame, obs, error_res, cmds = runner.step(dt)
        if error_res:
            final_error = error_res

    assert final_error is not None
    # Verify tracking error converges to <= 10 pixels (PS spec accuracy)
    assert final_error.error_magnitude <= 10.0
    assert final_error.running_rmse <= 15.0
