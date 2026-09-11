import sys
import argparse
import time
import os
import webbrowser
import cv2
import numpy as np
from config.schemas import SimConfig
from runner.simulator import SimulationRunner
from environment.motion import StraightLineMotion, CircularMotion

def launch_web_mission_control():
    """
    Launches the OptiTrack Web Server and opens http://localhost:8080/index.html
    automatically in the browser to show the full Mission Control Dashboard screen.
    """
    from server import main as run_server
    print("\n" + "=" * 75)
    print(" OPTITRACK — FSOC COARSE ALIGNMENT MISSION CONTROL")
    print(" Sat-to-UAV Coarse Alignment Simulator & Web Telemetry Server")
    print("=" * 75)
    print(" [+] Starting Live Server & Opening Web Dashboard in Browser...")
    print(" [+] URL: http://localhost:8080/index.html\n")
    
    # Auto-open browser after 0.8 seconds
    def _open_browser():
        time.sleep(0.8)
        webbrowser.open("http://localhost:8080/index.html")
    
    import threading
    threading.Thread(target=_open_browser, daemon=True).start()
    
    run_server()

def run_cli_mode():
    """Fallback lightweight OpenCV window mode."""
    print("=====================================================================")
    print(" PS 26169: CoarsePAT-Sim - Virtual Camera Tracking Simulator (CLI)")
    print("=====================================================================")
    print(" Controls: 's' - Straight Line | 'c' - Circular | 'q' / ESC - Quit")
    print("=====================================================================\n")

    config = SimConfig()
    runner = SimulationRunner(config)

    window_name = "PS 26169 CoarsePAT-Sim (CLI Mode)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    target_fps = 30.0
    dt = 1.0 / target_fps
    frame_count = 0
    start_time = time.time()
    last_fps_time = time.time()
    fps_display = 30.0

    try:
        while True:
            t0 = time.time()
            frame, obs, error_res, (pan_cmd, tilt_cmd), current_state = runner.step(dt)
            frame_count += 1

            img_bgr = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
            cx_center, cy_center = config.camera_res_x // 2, config.camera_res_y // 2

            # Boresight Crosshair
            cv2.line(img_bgr, (cx_center - 15, cy_center), (cx_center + 15, cy_center), (255, 0, 0), 1)
            cv2.line(img_bgr, (cx_center, cy_center - 15), (cx_center, cy_center + 15), (255, 0, 0), 1)
            cv2.circle(img_bgr, (cx_center, cy_center), 3, (255, 0, 0), -1)

            # Target & Centroid Overlays
            if obs and obs.is_valid:
                bx, by, bw, bh = obs.bbox
                cv2.rectangle(img_bgr, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
                det_x, det_y = int(round(obs.centroid_x)), int(round(obs.centroid_y))
                cv2.drawMarker(img_bgr, (det_x, det_y), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
                cv2.line(img_bgr, (cx_center, cy_center), (det_x, det_y), (0, 255, 255), 1)

            # Compute live FPS
            now = time.time()
            if now - last_fps_time >= 0.5:
                fps_display = frame_count / (now - start_time)
                last_fps_time = now

            # HUD
            err_str = f"{error_res.error_magnitude:.2f} px" if error_res else "N/A (LOST)"
            rmse_str = f"{error_res.running_rmse:.2f} px" if error_res else "N/A"
            hud_lines = [
                f"MODE: Virtual Sim (CLI Baseline)",
                f"PATTERN: {runner.config.motion_pattern.upper()}",
                f"CAM PAN/TILT: {frame.camera_pan:+.2f}deg / {frame.camera_tilt:+.2f}deg",
                f"CMD PAN/TILT: {pan_cmd:+.2f}deg/s / {tilt_cmd:+.2f}deg/s",
                f"TRACKING ERROR: {err_str}",
                f"RUNNING RMSE: {rmse_str}",
                f"FPS: {fps_display:.1f}"
            ]

            cv2.rectangle(img_bgr, (5, 5), (320, 155), (20, 20, 20), -1)
            cv2.rectangle(img_bgr, (5, 5), (320, 155), (100, 100, 100), 1)
            for idx, line in enumerate(hud_lines):
                color = (0, 255, 255) if "ERROR" in line else (0, 255, 0) if "FPS" in line else (255, 255, 255)
                cv2.putText(img_bgr, line, (10, 22 + idx * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

            cv2.imshow(window_name, img_bgr)
            elapsed = time.time() - t0
            sleep_ms = max(1, int((dt - elapsed) * 1000))
            key = cv2.waitKey(sleep_ms) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                runner.config.motion_pattern = "straight_line"
                runner.motion = StraightLineMotion(start_pos=runner.env.center(), speed_px_s=80.0, angle_deg=35.0)
                runner.error_calc.reset()
            elif key == ord('c'):
                runner.config.motion_pattern = "circular"
                runner.motion = CircularMotion(center_pos=runner.env.center(), radius_px=300.0, angular_speed_rad_s=0.5)
                runner.error_calc.reset()
    finally:
        cv2.destroyAllWindows()

def run_desktop_gui():
    """Desktop PySide6 GUI window mode."""
    try:
        from PySide6.QtWidgets import QApplication, QDialog
        from ui.main_window import CoarsePATMainWindow
        from ui.config_dialog import SimulationConfigDialog

        app = QApplication(sys.argv)
        config = SimConfig()
        window = CoarsePATMainWindow(config=config)
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"[Warning] Desktop GUI error ({e}). Launching Web Mission Control...")
        launch_web_mission_control()

def main():
    parser = argparse.ArgumentParser(description="OptiTrack — FSOC Coarse Alignment Mission Control")
    parser.add_argument("--gui", action="store_true", help="Run in PySide6 Desktop GUI mode")
    parser.add_argument("--cli", action="store_true", help="Run in lightweight OpenCV CLI mode")
    args = parser.parse_args()

    if args.gui:
        run_desktop_gui()
    elif args.cli:
        run_cli_mode()
    else:
        # Default: Launch Web Mission Control Dashboard Screen
        launch_web_mission_control()

if __name__ == "__main__":
    main()
