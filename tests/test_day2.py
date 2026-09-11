import os
import pytest
import numpy as np
from config.schemas import SimConfig
from environment.world import VirtualEnvironment
from environment.motion import StraightLineMotion, CircularMotion, FigureOf8Motion, RandomMotion
from disturbance.noise import ImageNoiseEngine
from disturbance.atmosphere import AtmosphericConditionEngine
from disturbance.jitter import CameraJitterEngine
from disturbance.platform import PlatformMotionEngine
from disturbance.engine import DisturbanceEngine
from tracking.state import TrackingState
from tracking.state_machine import TrackingStateMachine
from reacquisition.search import ReacquisitionSearchStrategy
from metrics.engine import PerformanceMeasurementEngine
from logging_system.logger import AutomaticLogger
from runner.simulator import SimulationRunner

def test_all_motion_patterns():
    center = (1000.0, 1000.0)
    line = StraightLineMotion(center, 50.0, 30.0)
    circle = CircularMotion(center, 200.0, 0.4)
    fig8 = FigureOf8Motion(center, 300.0, 200.0, 0.3)
    rand_walk = RandomMotion(center, random_seed=42)

    for m in [line, circle, fig8, rand_walk]:
        pos = m.get_position(1.0)
        assert len(pos) == 2
        assert 0 <= pos[0] <= 2000
        assert 0 <= pos[1] <= 2000

def test_disturbance_engine():
    img = np.zeros((480, 640), dtype=np.uint8)
    img[230:250, 310:330] = 255  # bright spot

    engine = DisturbanceEngine(random_seed=42)
    engine.enable_gaussian = True
    engine.enable_salt_pepper = True
    engine.atmospheric_condition = "Fog"
    engine.enable_jitter = True
    engine.enable_platform_motion = True

    degraded_img, meta = engine.process_frame(img, timestamp=1.0)
    assert degraded_img.shape == (480, 640)
    assert "atmosphere" in meta
    assert "gaussian_std" in meta
    assert "sp_density" in meta

def test_tracking_state_machine_transitions():
    sm = TrackingStateMachine(lock_debounce_count=2, loss_debounce_count=2)
    assert sm.state == TrackingState.SEARCHING

    # Step 1: First detection => BEACON_DETECTED
    from tracking.observation import CentroidObservation
    obs = CentroidObservation(320.0, 240.0, 0.9, (315, 235, 10, 10), 100.0, True)
    
    st1 = sm.update(obs, timestamp=0.1, dt=0.033)
    assert st1 == TrackingState.BEACON_DETECTED

    # Step 2: Second detection => ACQUISITION
    st2 = sm.update(obs, timestamp=0.133, dt=0.033)
    assert st2 == TrackingState.ACQUISITION

    # Step 3: Third detection => LOCKED
    st3 = sm.update(obs, timestamp=0.166, dt=0.033)
    assert st3 == TrackingState.LOCKED
    assert sm.is_locked()

    # Step 4: Missed detections => TARGET_LOST
    sm.update(None, timestamp=0.2, dt=0.033)
    st_lost = sm.update(None, timestamp=0.233, dt=0.033)
    assert st_lost in (TrackingState.TARGET_LOST, TrackingState.REACQUISITION)

def test_metrics_and_logger():
    metrics = PerformanceMeasurementEngine()
    metrics.record_frame(0, 0.0, TrackingState.LOCKED, None, 2.5)
    metrics.record_frame(1, 0.033, TrackingState.LOCKED, None, 2.8)

    summary = metrics.generate_summary(acquisition_time_sec=0.15)
    assert summary.total_frames == 2
    assert summary.acquisition_time_sec == 0.15

    logger = AutomaticLogger(log_dir="test_logs")
    csv_file, json_file = logger.save_run_logs(metrics.frame_records, summary)

    assert os.path.exists(csv_file)
    assert os.path.exists(json_file)

    # Cleanup test log files
    os.remove(csv_file)
    os.remove(json_file)
    os.rmdir("test_logs")

def test_full_day2_simulation_runner():
    config = SimConfig(motion_pattern="figure_of_8")
    runner = SimulationRunner(config)

    # Enable disturbances
    runner.disturbance_engine.enable_gaussian = True
    runner.disturbance_engine.gaussian_std_dev = 10.0

    dt = 1.0 / 30.0
    for _ in range(50):
        frame, obs, error_res, cmds, state = runner.step(dt)

    assert runner.metrics_engine.total_processed_frames == 50
    csv_file, json_file = runner.save_summary_logs()
    assert os.path.exists(csv_file)
    assert os.path.exists(json_file)
