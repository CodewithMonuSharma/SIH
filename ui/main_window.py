import sys
import time
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSlider, QGroupBox, QFormLayout, QFrame
)
from config.schemas import SimConfig
from runner.simulator import SimulationRunner
from environment.motion import StraightLineMotion, CircularMotion
from ui.widgets import WorldMapWidget, MetricCard

class CoarsePATMainWindow(QMainWindow):
    """
    Main PySide6 Scientific Desktop Dashboard for CoarsePAT-Sim (ISRO PS 26169).
    Combines 2D Virtual Environment Overview, 640x480 Live Camera Viewport,
    Real-Time Controls, and Telemetry Performance Dashboard.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ISRO PS 26169 — CoarsePAT-Sim | AI-Based Virtual Camera Tracking Simulator")
        self.resize(1280, 850)

        # Config & Runner
        self.config = SimConfig()
        self.runner = SimulationRunner(self.config)

        # Simulation loop timer
        self.is_running = True
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self._on_sim_tick)
        self.dt = 1.0 / 30.0

        # Frame rate tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 30.0

        # Setup UI
        self._init_styles()
        self._init_ui()

        # Start timer (30 FPS)
        self.sim_timer.start(int(self.dt * 1000))

    def _init_styles(self):
        """Dark Space Sci-Fi Theme stylesheet."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0d1117;
            }
            QWidget {
                color: #c9d1d9;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
                color: #8b949e;
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 14px;
                background-color: #161b22;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #58a6ff;
            }
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 6px;
                color: #c9d1d9;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #30363d;
                border-color: #8b949e;
            }
            QPushButton:pressed {
                background-color: #1f6feb;
                color: #ffffff;
            }
            QComboBox {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 6px 12px;
                color: #c9d1d9;
            }
            QSlider::groove:horizontal {
                border: 1px solid #30363d;
                height: 6px;
                background: #21262d;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #58a6ff;
                border: none;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # -------------------------------------------------------------
        # Header Bar
        # -------------------------------------------------------------
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px;")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(12, 4, 12, 4)

        title_label = QLabel("COARSE PAT SIMULATOR — ISRO PS 26169")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #58a6ff; letter-spacing: 1px;")

        self.status_badge = QLabel("TRACKING LOCKED")
        self.status_badge.setStyleSheet("""
            background-color: rgba(63, 185, 80, 0.15);
            color: #3fb950;
            border: 1px solid #3fb950;
            border-radius: 12px;
            padding: 4px 12px;
            font-weight: bold;
            font-size: 11px;
        """)

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_badge)
        main_layout.addWidget(header_frame)

        # -------------------------------------------------------------
        # Content Split Layout (Left: Visualizers, Right: Controls & Stats)
        # -------------------------------------------------------------
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        # Left Column: Dual Render Views
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        # World Mini-Map View
        self.world_map = WorldMapWidget(self.config.screen_width, self.config.screen_height)
        left_col.addWidget(self.world_map)

        # Camera Viewport Label
        cam_frame = QGroupBox("LIVE CAMERA VIEWPORT (640x480 CROP)")
        cam_layout = QVBoxLayout(cam_frame)
        cam_layout.setContentsMargins(8, 8, 8, 8)
        self.camera_feed_label = QLabel()
        self.camera_feed_label.setFixedSize(640, 480)
        self.camera_feed_label.setStyleSheet("background-color: #000000; border: 1px solid #30363d; border-radius: 4px;")
        cam_layout.addWidget(self.camera_feed_label, alignment=Qt.AlignCenter)
        left_col.addWidget(cam_frame)

        content_layout.addLayout(left_col, stretch=3)

        # Right Column: Controls & Metrics Dashboard
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        # 1. Interactive Control Panel
        controls_group = QGroupBox("SIMULATION CONTROLS & PARAMETERS")
        ctrl_layout = QFormLayout(controls_group)
        ctrl_layout.setContentsMargins(12, 12, 12, 12)
        ctrl_layout.setSpacing(10)

        # Buttons Row
        btn_row = QHBoxLayout()
        self.btn_toggle = QPushButton("PAUSE SIMULATION")
        self.btn_toggle.clicked.connect(self._toggle_simulation)
        self.btn_reset = QPushButton("RESET")
        self.btn_reset.clicked.connect(self._reset_simulation)
        btn_row.addWidget(self.btn_toggle)
        btn_row.addWidget(self.btn_reset)
        ctrl_layout.addRow(btn_row)

        # Pattern Dropdown
        self.combo_pattern = QComboBox()
        self.combo_pattern.addItems(["Straight Line", "Circular"])
        self.combo_pattern.currentTextChanged.connect(self._on_pattern_change)
        ctrl_layout.addRow("Motion Pattern:", self.combo_pattern)

        # Speed Slider
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(10, 150)
        self.slider_speed.setValue(40)
        self.lbl_speed_val = QLabel("40 px/s")
        self.slider_speed.valueChanged.connect(self._on_speed_change)
        ctrl_layout.addRow("Target Speed:", self._make_slider_row(self.slider_speed, self.lbl_speed_val))

        # Controller Kp Slider
        self.slider_kp = QSlider(Qt.Horizontal)
        self.slider_kp.setRange(10, 200)
        self.slider_kp.setValue(80)  # Kp = 8.0
        self.lbl_kp_val = QLabel("8.0")
        self.slider_kp.valueChanged.connect(self._on_kp_change)
        ctrl_layout.addRow("PAT Controller Kp:", self._make_slider_row(self.slider_kp, self.lbl_kp_val))

        # Max Speed Slider
        self.slider_max_speed = QSlider(Qt.Horizontal)
        self.slider_max_speed.setRange(10, 100)
        self.slider_max_speed.setValue(50)  # 5.0 deg/s
        self.lbl_max_speed_val = QLabel("5.0 deg/s")
        self.slider_max_speed.valueChanged.connect(self._on_max_speed_change)
        ctrl_layout.addRow("Max Pan/Tilt Speed:", self._make_slider_row(self.slider_max_speed, self.lbl_max_speed_val))

        right_col.addWidget(controls_group)

        # 2. Telemetry Cards Grid
        telemetry_group = QGroupBox("REAL-TIME TELEMETRY & METRICS")
        grid_layout = QGridLayout(telemetry_group)
        grid_layout.setContentsMargins(10, 10, 10, 10)
        grid_layout.setSpacing(8)

        self.card_error = MetricCard("Tracking Error", "0.0", "px")
        self.card_rmse = MetricCard("Running RMSE", "0.0", "px")
        self.card_pan = MetricCard("Camera Pan", "0.00", "deg")
        self.card_tilt = MetricCard("Camera Tilt", "0.00", "deg")
        self.card_pan_cmd = MetricCard("Pan Command", "0.00", "deg/s")
        self.card_tilt_cmd = MetricCard("Tilt Command", "0.00", "deg/s")
        self.card_gt = MetricCard("Target Ground Truth", "(1000, 1000)", "px")
        self.card_fps = MetricCard("Processing Rate", "30.0", "FPS")

        grid_layout.addWidget(self.card_error, 0, 0)
        grid_layout.addWidget(self.card_rmse, 0, 1)
        grid_layout.addWidget(self.card_pan, 1, 0)
        grid_layout.addWidget(self.card_tilt, 1, 1)
        grid_layout.addWidget(self.card_pan_cmd, 2, 0)
        grid_layout.addWidget(self.card_tilt_cmd, 2, 1)
        grid_layout.addWidget(self.card_gt, 3, 0)
        grid_layout.addWidget(self.card_fps, 3, 1)

        right_col.addWidget(telemetry_group)
        right_col.addStretch()

        content_layout.addLayout(right_col, stretch=2)
        main_layout.addLayout(content_layout)

    def _make_slider_row(self, slider: QSlider, val_label: QLabel) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider)
        val_label.setFixedWidth(65)
        layout.addWidget(val_label)
        return container

    def _toggle_simulation(self):
        if self.is_running:
            self.sim_timer.stop()
            self.is_running = False
            self.btn_toggle.setText("RESUME SIMULATION")
            self.btn_toggle.setStyleSheet("background-color: #1f6feb; color: #ffffff;")
        else:
            self.sim_timer.start(int(self.dt * 1000))
            self.is_running = True
            self.btn_toggle.setText("PAUSE SIMULATION")
            self.btn_toggle.setStyleSheet("background-color: #21262d; color: #c9d1d9;")

    def _reset_simulation(self):
        self.runner = SimulationRunner(self.config)
        self.frame_count = 0
        self.start_time = time.time()
        self._on_sim_tick()

    def _on_pattern_change(self, text: str):
        center = self.runner.env.center()
        if text == "Circular":
            self.config.motion_pattern = "circular"
            self.runner.motion = CircularMotion(center_pos=center, radius_px=300.0, angular_speed_rad_s=0.5)
        else:
            self.config.motion_pattern = "straight_line"
            self.runner.motion = StraightLineMotion(start_pos=center, speed_px_s=self.config.motion_speed_px_s, angle_deg=35.0)
        self.runner.error_calc.reset()

    def _on_speed_change(self, val: int):
        self.config.motion_speed_px_s = float(val)
        self.lbl_speed_val.setText(f"{val} px/s")
        if isinstance(self.runner.motion, StraightLineMotion):
            self.runner.motion.speed = float(val)
            self.runner.motion.vx = self.runner.motion.speed * np.cos(self.runner.motion.angle_rad)
            self.runner.motion.vy = self.runner.motion.speed * np.sin(self.runner.motion.angle_rad)

    def _on_kp_change(self, val: int):
        kp_float = val / 10.0
        self.lbl_kp_val.setText(f"{kp_float:.1f}")
        self.runner.controller.kp = kp_float

    def _on_max_speed_change(self, val: int):
        speed_float = val / 10.0
        self.lbl_max_speed_val.setText(f"{speed_float:.1f} deg/s")
        self.config.max_pan_speed_deg_s = speed_float
        self.config.max_tilt_speed_deg_s = speed_float
        self.runner.camera.pan_tilt.max_pan_speed = speed_float
        self.runner.camera.pan_tilt.max_tilt_speed = speed_float
        self.runner.controller.max_pan_speed = speed_float
        self.runner.controller.max_tilt_speed = speed_float

    def _on_sim_tick(self):
        """Simulation tick callback called by QTimer at 30 FPS."""
        frame, obs, error_res, (pan_cmd, tilt_cmd) = self.runner.step(self.dt)
        self.frame_count += 1

        # 1. Update 2D World Mini-Map
        crop_rect = self.runner.camera.viewport.get_crop_rect(frame.camera_pan, frame.camera_tilt)
        target_gt = (frame.ground_truth_x, frame.ground_truth_y)
        self.world_map.update_state(target_gt, crop_rect)

        # 2. Render Live Camera Frame Overlays
        img_bgr = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
        cx_center, cy_center = self.config.camera_res_x // 2, self.config.camera_res_y // 2

        # Boresight Crosshair
        cv2.line(img_bgr, (cx_center - 15, cy_center), (cx_center + 15, cy_center), (255, 0, 0), 1)
        cv2.line(img_bgr, (cx_center, cy_center - 15), (cx_center, cy_center + 15), (255, 0, 0), 1)
        cv2.circle(img_bgr, (cx_center, cy_center), 3, (255, 0, 0), -1)

        # Detected Beacon Overlay
        if obs and obs.is_valid:
            bx, by, bw, bh = obs.bbox
            cv2.rectangle(img_bgr, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
            det_x, det_y = int(round(obs.centroid_x)), int(round(obs.centroid_y))
            cv2.drawMarker(img_bgr, (det_x, det_y), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
            cv2.line(img_bgr, (cx_center, cy_center), (det_x, det_y), (0, 255, 255), 1)

        # Convert OpenCV BGR image to QPixmap for QLabel
        h, w, ch = img_bgr.shape
        bytes_per_line = ch * w
        q_img = QImage(img_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.camera_feed_label.setPixmap(QPixmap.fromImage(q_img))

        # 3. Update Telemetry Cards
        if error_res:
            err_val = error_res.error_magnitude
            err_color = "#3fb950" if err_val <= 10.0 else "#d29922" if err_val <= 20.0 else "#f85149"
            self.card_error.set_value(f"{err_val:.2f}", err_color)
            self.card_rmse.set_value(f"{error_res.running_rmse:.2f}", "#58a6ff")
            self.status_badge.setText("TRACKING LOCKED")
            self.status_badge.setStyleSheet("background-color: rgba(63, 185, 80, 0.15); color: #3fb950; border: 1px solid #3fb950; border-radius: 12px; padding: 4px 12px; font-weight: bold;")
        else:
            self.card_error.set_value("N/A", "#f85149")
            self.card_rmse.set_value("N/A", "#f85149")
            self.status_badge.setText("TARGET LOST")
            self.status_badge.setStyleSheet("background-color: rgba(248, 81, 73, 0.15); color: #f85149; border: 1px solid #f85149; border-radius: 12px; padding: 4px 12px; font-weight: bold;")

        self.card_pan.set_value(f"{frame.camera_pan:+.2f}")
        self.card_tilt.set_value(f"{frame.camera_tilt:+.2f}")
        self.card_pan_cmd.set_value(f"{pan_cmd:+.2f}")
        self.card_tilt_cmd.set_value(f"{tilt_cmd:+.2f}")
        self.card_gt.set_value(f"({frame.ground_truth_x:.0f}, {frame.ground_truth_y:.0f})")

        # Update FPS
        now = time.time()
        elapsed = now - self.start_time
        if elapsed >= 0.5:
            self.fps = self.frame_count / elapsed
            self.card_fps.set_value(f"{self.fps:.1f}", "#3fb950" if self.fps >= 20.0 else "#f85149")

def main():
    app = QApplication(sys.argv)
    window = CoarsePATMainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    main()
