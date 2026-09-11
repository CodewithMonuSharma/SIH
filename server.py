"""
OptiTrack — FSOC Coarse Alignment Mission Control
Real-Time Python Simulation Backend & Web Streaming Server
Streams OpenCV Frames, Sensor Telemetry, and 2-way Hardware Control via SSE & REST API.
"""

import http.server
import socketserver
import webbrowser
import os
import sys
import json
import math
import time
import base64
import threading
import cv2
import numpy as np
from typing import Optional, Dict, Any

from config.schemas import SimConfig
from runner.simulator import SimulationRunner
from environment.motion import (
    StraightLineMotion, CircularMotion, FigureOf8Motion, RandomMotion,
    SpiralMotion, SinusoidMotion
)
from tracking.state import TrackingState

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts all NumPy scalar/array types to native Python."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _safe(v):
    """Convert a single value to a JSON-safe native Python type."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


class SimulationBridge:
    """
    Thread-safe bridge between Python SimulationRunner and Web Dashboard frontend.
    """
    def __init__(self):
        self.config = SimConfig()
        self.runner = SimulationRunner(self.config)
        self.is_running = True
        self.dt = 1.0 / 30.0
        self.lock = threading.Lock()

        # Telemetry State
        self.last_frame_b64: str = ""
        self.last_payload: Dict[str, Any] = {}
        self.fps_display = 30.0
        self.fps_frame_count = 0      # resets every 0.3s for FPS calc
        self._total_frame_count = 0   # never resets — used for telemetry logging
        self.fps_window_start = time.time()
        self.telemetry_history = []

        # Start simulation worker thread
        self.thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.thread.start()

    def _sim_loop(self):
        while True:
            t0 = time.time()
            if self.is_running:
                with self.lock:
                    try:
                        frame, obs, error_res, (pan_cmd, tilt_cmd), current_state = self.runner.step(self.dt)
                        self.fps_frame_count += 1
                        self._total_frame_count += 1

                        # Compute target ground truth & gimbal center in world coords
                        target_pos = self.runner.motion.get_position(self.runner.sim_time)
                        crop_x1, crop_y1, crop_x2, crop_y2 = self.runner.camera.viewport.get_crop_rect(
                            self.runner.camera.pan_tilt.pan_deg, self.runner.camera.pan_tilt.tilt_deg
                        )
                        gimbal_x = (crop_x1 + crop_x2) / 2.0
                        gimbal_y = (crop_y1 + crop_y2) / 2.0

                        # Encode frame image to JPEG Base64
                        img = frame.image
                        if len(img.shape) == 2:
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        else:
                            img_bgr = img

                        _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        frame_b64 = base64.b64encode(buf).decode('utf-8')

                        # FPS calculation (resets every 0.3s)
                        now = time.time()
                        if now - self.fps_window_start >= 0.3:
                            self.fps_display = self.fps_frame_count / max(0.001, (now - self.fps_window_start))
                            self.fps_frame_count = 0
                            self.fps_window_start = now

                        obs_dict = None
                        if obs and obs.is_valid:
                            obs_dict = {
                                "cx": round(float(obs.centroid_x), 2),
                                "cy": round(float(obs.centroid_y), 2),
                                "bbox": [int(v) for v in obs.bbox] if obs.bbox is not None else None,
                                "is_valid": True
                            }

                        err_val = round(float(error_res.error_magnitude), 2) if error_res else 0.0
                        rmse_val = round(float(error_res.running_rmse), 2) if error_res else 0.0

                        # Pull acquisition/reacquisition timing from state machine
                        acq_time = self.runner.state_machine.acquisition_time_sec
                        reacq_time = self.runner.state_machine.last_reacquisition_time_sec

                        # Lock retention & loss rates from metrics engine
                        metrics = self.runner.metrics_engine
                        total_f = max(1, metrics.total_processed_frames)
                        lock_retention = round(metrics.locked_frames / total_f * 100.0, 2)
                        loss_rate = round(metrics.lost_frames / total_f * 100.0, 2)

                        # Determine state label for frontend display
                        state_labels = {
                            "INITIALIZATION": "INIT",
                            "SEARCHING": "SEARCHING",
                            "BEACON_DETECTED": "DETECTED",
                            "ACQUISITION": "ACQUIRING",
                            "LOCKED": "LOCKED",
                            "TARGET_LOST": "TARGET LOST",
                            "REACQUISITION": "RE-ACQUIRING",
                        }
                        state_label = state_labels.get(current_state.name, current_state.name)

                        self.last_payload = {
                            "time": round(float(self.runner.sim_time), 2),
                            "target_pos": [round(float(target_pos[0]), 1), round(float(target_pos[1]), 1)],
                            "gimbal_pos": [round(float(gimbal_x), 1), round(float(gimbal_y), 1)],
                            "pan_deg": round(float(frame.camera_pan), 2),
                            "tilt_deg": round(float(frame.camera_tilt), 2),
                            "pan_cmd": round(float(pan_cmd), 2),
                            "tilt_cmd": round(float(tilt_cmd), 2),
                            "error_px": err_val,
                            "rmse_px": rmse_val,
                            "fps": round(float(self.fps_display), 1),
                            "state": str(current_state.name),
                            "state_label": state_label,
                            "is_locked": bool(current_state in (TrackingState.LOCKED, TrackingState.ACQUISITION)),
                            "acq_time": round(float(acq_time), 3) if acq_time is not None else None,
                            "reacq_time": round(float(reacq_time), 3) if reacq_time is not None else None,
                            "lock_retention": lock_retention,
                            "loss_rate": loss_rate,
                            "obs": obs_dict,
                            "frame_jpg": frame_b64
                        }

                        # Log every 30 total frames for telemetry history (using non-resetting counter)
                        if self._total_frame_count % 30 == 0:
                            record = {
                                "time": self.last_payload["time"],
                                "tgtX": self.last_payload["target_pos"][0],
                                "tgtY": self.last_payload["target_pos"][1],
                                "gimbalX": self.last_payload["gimbal_pos"][0],
                                "gimbalY": self.last_payload["gimbal_pos"][1],
                                "errorPx": err_val,
                                "fps": self.last_payload["fps"],
                                "state": state_label,
                                "lockRetention": lock_retention,
                            }
                            self.telemetry_history.append(record)
                            if len(self.telemetry_history) > 500:
                                self.telemetry_history.pop(0)

                    except Exception as e:
                        import traceback
                        print(f"[Simulation Error]: {e}")
                        traceback.print_exc()

            elapsed = time.time() - t0
            sleep_time = max(0.002, self.dt - elapsed)
            time.sleep(sleep_time)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Returns a full live performance metrics summary as a dict."""
        with self.lock:
            try:
                summary = self.runner.metrics_engine.generate_summary(
                    acquisition_time_sec=self.runner.state_machine.acquisition_time_sec,
                    reacquisition_time_sec=self.runner.state_machine.last_reacquisition_time_sec
                )
                return {
                    "status": "ok",
                    "sim_time": round(self.runner.sim_time, 2),
                    "fps": summary.fps,
                    "total_frames": summary.total_frames,
                    "acquisition_time_sec": summary.acquisition_time_sec,
                    "reacquisition_time_sec": summary.reacquisition_time_sec,
                    "mean_error_px": summary.mean_tracking_error_px,
                    "max_error_px": summary.max_tracking_error_px,
                    "rmse_px": summary.rmse_px,
                    "loss_percent": summary.target_loss_percent,
                    "lock_retention_percent": summary.lock_retention_rate_percent,
                    "avg_latency_ms": summary.avg_processing_latency_ms,
                    "telemetry_rows": len(self.telemetry_history),
                    "telemetry_history": self.telemetry_history[-100:],  # last 100 records
                }
            except Exception as e:
                return {"status": "error", "message": str(e)}

    def handle_command(self, cmd_data: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            action = cmd_data.get("action", "")

            if action == "run":
                self.is_running = True
                return {"status": "ok", "is_running": True}

            elif action == "pause":
                self.is_running = False
                return {"status": "ok", "is_running": False}

            elif action == "reset":
                self.runner.sim_time = 0.0
                self.runner.error_calc.reset()
                self.runner.state_machine.__init__()   # full SM reset
                self.runner.metrics_engine.__init__()  # reset metrics
                self._total_frame_count = 0
                self.fps_frame_count = 0
                self.fps_window_start = time.time()
                self.telemetry_history.clear()
                return {"status": "ok", "reset": True}

            elif action == "set_pattern":
                pattern = cmd_data.get("pattern", "straight")
                center = self.runner.env.center()
                spd = self.config.motion_speed_px_s
                if pattern == "straight":
                    self.runner.motion = StraightLineMotion(start_pos=center, speed_px_s=spd, angle_deg=35.0)
                elif pattern == "circular":
                    self.runner.motion = CircularMotion(center_pos=center, radius_px=300.0,
                                                         angular_speed_rad_s=0.5 * (spd / 80.0))
                elif pattern in ("figure_of_8", "figure8"):
                    self.runner.motion = FigureOf8Motion(center_pos=center, scale_x=400.0, scale_y=250.0,
                                                          angular_speed_rad_s=0.3 * (spd / 80.0))
                elif pattern == "random":
                    self.runner.motion = RandomMotion(start_pos=center,
                                                       bounds=(self.config.screen_width, self.config.screen_height),
                                                       step_size_px=50.0 * (spd / 80.0))
                elif pattern == "spiral":
                    self.runner.motion = SpiralMotion(center_pos=center,
                                                       expansion_rate=15.0 * (spd / 80.0),
                                                       angular_speed_rad_s=0.5 * (spd / 80.0))
                elif pattern == "sinusoid":
                    self.runner.motion = SinusoidMotion(start_pos=center, speed_px_s=spd)
                return {"status": "ok", "pattern": pattern}

            elif action == "set_speed":
                speed = float(cmd_data.get("speed", 80.0))
                self.config.motion_speed_px_s = speed
                # Update the live motion object's speed without replacing it
                m = self.runner.motion
                if isinstance(m, StraightLineMotion):
                    m.speed = speed
                    m.vx = speed * math.cos(m.angle_rad)
                    m.vy = speed * math.sin(m.angle_rad)
                elif isinstance(m, CircularMotion):
                    m.omega = 0.5 * (speed / 80.0)
                elif isinstance(m, FigureOf8Motion):
                    m.omega = 0.3 * (speed / 80.0)
                elif isinstance(m, SinusoidMotion):
                    m.speed = speed
                elif isinstance(m, SpiralMotion):
                    m.rate = 15.0 * (speed / 80.0)
                    m.omega = 0.5 * (speed / 80.0)
                return {"status": "ok", "speed": speed}

            elif action == "randomize":
                rx = float(np.random.uniform(200, self.config.screen_width - 200))
                ry = float(np.random.uniform(200, self.config.screen_height - 200))
                angle = float(np.random.uniform(0, 360))
                self.runner.motion = StraightLineMotion(
                    start_pos=(rx, ry),
                    speed_px_s=self.config.motion_speed_px_s,
                    angle_deg=angle
                )
                return {"status": "ok", "target_pos": [rx, ry]}

            elif action == "set_atmosphere":
                atmos_map = {
                    "clear": "Clear",
                    "haze": "Haze",
                    "fog": "Fog",
                    "rain": "Rain",
                    "night": "Night"
                }
                atmos = cmd_data.get("atmosphere", "clear").lower()
                target_atmos = atmos_map.get(atmos, "Clear")
                self.runner.disturbance_engine.atmospheric_condition = target_atmos
                return {"status": "ok", "atmosphere": target_atmos}

            elif action == "set_noise":
                self.runner.disturbance_engine.enable_gaussian = bool(cmd_data.get("gaussian", False))
                self.runner.disturbance_engine.enable_salt_pepper = bool(cmd_data.get("salt_pepper", False))
                self.runner.disturbance_engine.enable_poisson = bool(cmd_data.get("poisson", False))
                self.runner.disturbance_engine.enable_jitter = bool(cmd_data.get("jitter", False))
                self.runner.disturbance_engine.enable_platform_motion = bool(cmd_data.get("platform", False))
                return {"status": "ok"}

            elif action == "set_scene_size":
                w = int(cmd_data.get("width", 2000))
                h = int(cmd_data.get("height", 2000))
                self.config.screen_width = w
                self.config.screen_height = h
                if hasattr(self.runner, 'env'):
                    self.runner.env.width = w
                    self.runner.env.height = h
                return {"status": "ok", "width": w, "height": h}

            elif action == "set_target_coords":
                x = float(cmd_data.get("x", 1000.0))
                y = float(cmd_data.get("y", 1000.0))
                self.runner.motion = StraightLineMotion(
                    start_pos=(x, y),
                    speed_px_s=self.config.motion_speed_px_s,
                    angle_deg=35.0
                )
                return {"status": "ok", "x": x, "y": y}

            elif action == "set_target_size":
                size = int(round(float(cmd_data.get("size", 12.0))))
                self.config.target_size_px = size
                if hasattr(self.runner, 'beacon_renderer'):
                    self.runner.beacon_renderer.size_px = size
                return {"status": "ok", "size": size}

            elif action == "set_target_geometry":
                geom = cmd_data.get("geometry", "square")
                if hasattr(self.runner, 'beacon_renderer'):
                    self.runner.beacon_renderer.shape = "circle" if geom == "circle" else "square"
                return {"status": "ok", "geometry": geom}

            elif action == "set_pan_speed":
                pan_spd = float(cmd_data.get("pan_speed", 7.5))
                self.config.max_pan_speed_deg_s = pan_spd
                if hasattr(self.runner, 'camera') and hasattr(self.runner.camera, 'pan_tilt'):
                    self.runner.camera.pan_tilt.max_pan_speed = pan_spd
                return {"status": "ok", "pan_speed": pan_spd}

            elif action == "set_tilt_speed":
                tilt_spd = float(cmd_data.get("tilt_speed", 6.8))
                self.config.max_tilt_speed_deg_s = tilt_spd
                if hasattr(self.runner, 'camera') and hasattr(self.runner.camera, 'pan_tilt'):
                    self.runner.camera.pan_tilt.max_tilt_speed = tilt_spd
                return {"status": "ok", "tilt_speed": tilt_spd}

            elif action == "set_update_rate":
                hz = float(cmd_data.get("rate", 30.0))
                hz = max(20.0, min(120.0, hz))
                self.dt = 1.0 / hz
                return {"status": "ok", "rate": hz}

            elif action == "set_camera_type":
                cam_type = cmd_data.get("camera_type", "ir")
                return {"status": "ok", "camera_type": cam_type}

            elif action == "set_camera_res":
                res = cmd_data.get("res", "640x480")
                parts = res.split("x")
                if len(parts) == 2:
                    cw, ch = int(parts[0]), int(parts[1])
                    if hasattr(self.runner, 'camera') and hasattr(self.runner.camera, 'viewport'):
                        self.runner.camera.viewport.crop_w = cw
                        self.runner.camera.viewport.crop_h = ch
                return {"status": "ok", "res": res}

            elif action == "set_noise_params":
                if "std_dev" in cmd_data:
                    self.runner.disturbance_engine.gaussian_std_dev = float(cmd_data["std_dev"])
                if "jitter" in cmd_data:
                    val = float(cmd_data["jitter"])
                    self.runner.disturbance_engine.jitter_max_px = val
                    self.runner.disturbance_engine.enable_jitter = (val > 0)
                if "platform_mag" in cmd_data:
                    val = float(cmd_data["platform_mag"])
                    self.runner.disturbance_engine.platform_max_px = val
                    self.runner.disturbance_engine.enable_platform_motion = (val > 0)
                return {"status": "ok"}

            elif action == "set_platform_type":
                p_type = cmd_data.get("platform_type", "linear").title()
                if hasattr(self.runner.disturbance_engine, 'platform_engine'):
                    self.runner.disturbance_engine.platform_engine.pattern = p_type
                return {"status": "ok", "platform_type": p_type}

            elif action == "set_init_cam_pos":
                pos_mode = cmd_data.get("pos_mode", "center")
                if pos_mode == "center":
                    self.runner.camera.pan_tilt.pan_deg = 0.0
                    self.runner.camera.pan_tilt.tilt_deg = 0.0
                elif pos_mode == "topleft":
                    self.runner.camera.pan_tilt.pan_deg = -10.0
                    self.runner.camera.pan_tilt.tilt_deg = -10.0
                elif pos_mode == "offset":
                    self.runner.camera.pan_tilt.pan_deg = 5.0
                    self.runner.camera.pan_tilt.tilt_deg = 5.0
                return {"status": "ok", "pos_mode": pos_mode}

            elif action == "export_report":
                return self.get_metrics_summary()

            return {"status": "error", "message": f"Unknown action: {action}"}


bridge = SimulationBridge()


class OptiTrackHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/stream':
            # Server-Sent Events (SSE) Real-Time Data Stream
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self._send_cors_headers()
            self.end_headers()

            try:
                while True:
                    if bridge.last_payload:
                        data_str = json.dumps(bridge.last_payload, cls=NumpyEncoder)
                        msg = f"data: {data_str}\n\n"
                        self.wfile.write(msg.encode('utf-8'))
                        self.wfile.flush()
                    time.sleep(0.033)  # ~30 FPS
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(bridge.last_payload, cls=NumpyEncoder).encode('utf-8'))
            return

        elif self.path == '/api/metrics':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            summary = bridge.get_metrics_summary()
            self.wfile.write(json.dumps(summary, cls=NumpyEncoder).encode('utf-8'))
            return

        return super().do_GET()

    def do_POST(self):
        if self.path == '/api/control':
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len)
            try:
                cmd = json.loads(post_body.decode('utf-8'))
                res = bridge.handle_command(cmd)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(res, cls=NumpyEncoder).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress per-request log spam for static files; only show API calls
        if '/api/' in (args[0] if args else ''):
            super().log_message(format, *args)


def main():
    os.chdir(DIRECTORY)
    # Enable SO_REUSEADDR
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), OptiTrackHTTPHandler) as httpd:
        url = f"http://localhost:{PORT}/index.html"
        print("=" * 75)
        print(" OPTITRACK — FSOC COARSE ALIGNMENT MISSION CONTROL (v2.4)")
        print(" Python Live Simulation Bridge + Web Stream Server")
        print("=" * 75)
        print(f" [+] Web Dashboard: {url}")
        print(f" [+] SSE Stream:    http://localhost:{PORT}/api/stream")
        print(f" [+] Metrics API:   http://localhost:{PORT}/api/metrics")
        print(f" [+] Control API:   POST http://localhost:{PORT}/api/control")
        print(" [+] Press Ctrl+C in terminal to stop.")
        print("=" * 75)

        try:
            webbrowser.open(url)
        except Exception:
            pass

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[!] Shutting down OptiTrack server.")
            httpd.server_close()


if __name__ == "__main__":
    main()
