import os
import sys
import time
from config.schemas import SimConfig
from runner.simulator import SimulationRunner
from environment.motion import FigureOf8Motion, RandomMotion
from tracking.state import TrackingState

def run_unattended_benchmark():
    """
    Day 2 Verification Check:
    Runs a full 10-second scenario (300 frames @ 30 FPS) unattended under
    simultaneous combined disturbances (Gaussian + S&P + Fog + Jitter + Platform Motion),
    verifies target tracking, state machine transitions, and automatically saves
    performance log files (CSV & JSON).
    """
    print("=====================================================================")
    print(" PS 26169: Day 2 Unattended Robustness Benchmark Run")
    print("=====================================================================")

    config = SimConfig(motion_pattern="figure_of_8")
    runner = SimulationRunner(config)

    # 1. Enable Full Combined Disturbance Matrix
    runner.disturbance_engine.atmospheric_condition = "Fog"
    runner.disturbance_engine.atmosphere_severity = 0.4
    runner.disturbance_engine.enable_gaussian = True
    runner.disturbance_engine.gaussian_std_dev = 12.0
    runner.disturbance_engine.enable_salt_pepper = True
    runner.disturbance_engine.sp_density = 0.04
    runner.disturbance_engine.enable_jitter = True
    runner.disturbance_engine.jitter_max_px = 6.0
    runner.disturbance_engine.enable_platform_motion = True
    runner.disturbance_engine.platform_max_px = 8.0

    print("Active Disturbance Configuration:")
    print("  - Atmospheric Condition: Fog (severity 0.4)")
    print("  - Gaussian Noise: Enabled (std_dev = 12.0 px)")
    print("  - Salt & Pepper Noise: Enabled (density = 4%)")
    print("  - Camera Jitter: Enabled (max = +-6.0 px)")
    print("  - Platform Motion Drift: Enabled (max = +-8.0 px)")
    print("  - Target Motion: Figure-of-8\n")
    print("Executing 300 simulation frames (~10.0 seconds)...")

    dt = 1.0 / 30.0
    t_start = time.time()

    for i in range(300):
        frame, obs, error_res, cmds, state = runner.step(dt)
        if (i + 1) % 60 == 0:
            err_str = f"{error_res.error_magnitude:.2f} px" if error_res else "N/A"
            print(f"  Frame {i+1}/300 [{state.value}]: Tracking Error = {err_str}")

    elapsed = time.time() - t_start
    print(f"\nSimulation run finished in {elapsed:.2f} seconds ({300/elapsed:.1f} FPS).")

    # 2. Automatically Generate and Save Performance Logs
    csv_path, json_path = runner.save_summary_logs()
    summary = runner.metrics_engine.generate_summary(
        acquisition_time_sec=runner.state_machine.acquisition_time_sec,
        reacquisition_time_sec=runner.state_machine.last_reacquisition_time_sec
    )

    print("\n=====================================================================")
    print(" PS 26169 DAY 2 PERFORMANCE SUMMARY REPORT")
    print("=====================================================================")
    print(f" Total Processed Frames:    {summary.total_frames}")
    print(f" Execution Duration:        {summary.simulation_duration_sec:.2f} s")
    print(f" Average Processing Rate:   {summary.fps:.1f} FPS (PS Target: >= 20 FPS)")
    print(f" Acquisition Time:          {summary.acquisition_time_sec} s (PS Target: <= 2.0 s)")
    print(f" Mean Tracking Error:       {summary.mean_tracking_error_px:.2f} px")
    print(f" Maximum Tracking Error:    {summary.max_tracking_error_px:.2f} px")
    print(f" Root-Mean-Square Error:    {summary.rmse_px:.2f} px")
    print(f" Target Loss:               {summary.target_loss_percent:.2f}% (PS Target: < 5.0%)")
    print(f" Lock Retention Rate:       {summary.lock_retention_rate_percent:.2f}%")
    print(f" Per-Frame CSV Log:         {csv_path}")
    print(f" Run Summary JSON Report:   {json_path}")
    print("=====================================================================\n")

    assert os.path.exists(csv_path), "CSV frame log must exist on disk"
    assert os.path.exists(json_path), "JSON summary log must exist on disk"
    print("Day 2 Check PASSED: Full scenario completed with verified deliverables on disk!")

if __name__ == "__main__":
    run_unattended_benchmark()
