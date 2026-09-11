import sys
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFrame, QScrollArea, QTabWidget, QMessageBox
)
from config.schemas import SimConfig

class SimulationConfigDialog(QDialog):
    """
    Futuristic Mission Control Parameter Configuration Modal for CoarsePAT-Sim (ISRO PS-26169).
    Presents a detailed configuration table allowing the operator to input all optical,
    target, kinematic, camera, and disturbance parameters before starting simulation.
    """
    def __init__(self, current_config: SimConfig, disturbance_engine=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MISSION PARAMETER CONFIGURATION — ISRO PS-26169")
        self.resize(860, 680)
        self.setMinimumSize(780, 600)
        self.config = current_config
        self.dist_engine = disturbance_engine
        self.start_immediately = False

        self._init_styles()
        self._init_ui()
        self._load_from_config()

    def _init_styles(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #060a12;
            }
            QWidget {
                color: #c9d1d9;
                font-family: 'Segoe UI', -apple-system, sans-serif;
            }
            QFrame.panel-card {
                background-color: #0b111e;
                border: 1px solid #142036;
                border-radius: 8px;
            }
            QLabel.section-hdr {
                font-size: 11px;
                font-weight: bold;
                color: #00f0ff;
                letter-spacing: 0.5px;
                font-family: 'Segoe UI', Arial;
            }
            QLabel.field-lbl {
                font-size: 10px;
                color: #94a3b8;
                font-weight: 600;
            }
            QLabel.field-unit {
                font-size: 9px;
                color: #64748b;
                font-family: Consolas, monospace;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #0e1726;
                border: 1px solid #1e293b;
                border-radius: 4px;
                padding: 5px 8px;
                color: #00f0ff;
                font-family: 'Consolas', monospace;
                font-size: 10px;
                font-weight: bold;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #00f0ff;
            }
            QComboBox QAbstractItemView {
                background-color: #0e1726;
                border: 1px solid #1e293b;
                selection-background-color: #1e293b;
                color: #00f0ff;
            }
            QCheckBox {
                spacing: 8px;
                color: #94a3b8;
                font-size: 10px;
                background-color: #080d1a;
                border: 1px solid #142036;
                border-radius: 4px;
                padding: 6px 10px;
            }
            QCheckBox:checked {
                background-color: #0a1f2e;
                border: 1px solid #00f0ff;
                color: #00f0ff;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                border-radius: 2px;
                border: 1px solid #334155;
                background-color: #0e1726;
            }
            QCheckBox::indicator:checked {
                background-color: #00f0ff;
                border-color: #00f0ff;
            }
            QPushButton.launch-btn {
                background-color: #00f0ff;
                border: 1px solid #00f0ff;
                border-radius: 6px;
                color: #060a12;
                padding: 8px 20px;
                font-size: 12px;
                font-weight: 800;
                font-family: 'Consolas', monospace;
                letter-spacing: 0.5px;
            }
            QPushButton.launch-btn:hover {
                background-color: #38bdf8;
                border-color: #ffffff;
                color: #000000;
            }
            QPushButton.secondary-btn {
                background-color: #0e1726;
                border: 1px solid #1e293b;
                border-radius: 6px;
                color: #94a3b8;
                padding: 8px 16px;
                font-size: 10px;
                font-weight: bold;
                font-family: 'Consolas', monospace;
            }
            QPushButton.secondary-btn:hover {
                background-color: #1e293b;
                border-color: #00f0ff;
                color: #ffffff;
            }
            QPushButton.preset-pill {
                background-color: #080d1a;
                border: 1px solid #1e293b;
                border-radius: 4px;
                color: #38bdf8;
                font-family: 'Consolas', monospace;
                font-size: 9px;
                font-weight: bold;
                padding: 4px 8px;
            }
            QPushButton.preset-pill:hover {
                background-color: #0d2838;
                border-color: #00f0ff;
            }
        """)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(12)

        # -------------------------------------------------------------
        # 1. Header Banner
        # -------------------------------------------------------------
        hdr_frame = QFrame()
        hdr_frame.setProperty("class", "panel-card")
        hdr_frame.setStyleSheet("background-color: #080e1a; border: 1px solid #142036; border-radius: 8px;")
        hdr_layout = QHBoxLayout(hdr_frame)
        hdr_layout.setContentsMargins(14, 10, 14, 10)
        hdr_layout.setSpacing(10)

        pill = QFrame()
        pill.setFixedSize(4, 34)
        pill.setStyleSheet("background-color: #00f0ff; border-radius: 2px;")
        hdr_layout.addWidget(pill)

        hdr_vbox = QVBoxLayout()
        hdr_vbox.setSpacing(2)

        title = QLabel("SIMULATION CONFIGURATION TABLE & PARAMETER SETUP")
        title.setStyleSheet("font-size: 13px; font-weight: 800; color: #ffffff; letter-spacing: 0.5px;")
        hdr_vbox.addWidget(title)

        sub = QLabel("ISRO PS-26169 CoarsePAT-Sim  •  Configure Target Kinematics, Optics & Disturbances before start.")
        sub.setStyleSheet("font-size: 9px; color: #64748b; font-family: Consolas, monospace;")
        hdr_vbox.addWidget(sub)

        hdr_layout.addLayout(hdr_vbox)
        hdr_layout.addStretch()

        # Preset shortcut buttons on header
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        lbl_p = QLabel("PRESETS:")
        lbl_p.setStyleSheet("font-size: 8.5px; color: #64748b; font-weight: bold;")
        preset_row.addWidget(lbl_p)

        btn_p_geo = QPushButton("GEO_NOMINAL")
        btn_p_geo.setProperty("class", "preset-pill")
        btn_p_geo.clicked.connect(self._load_geo_preset)

        btn_p_leo = QPushButton("LEO_SLEW_HI")
        btn_p_leo.setProperty("class", "preset-pill")
        btn_p_leo.clicked.connect(self._load_leo_preset)

        btn_p_day2 = QPushButton("DAY_2_DISTURB")
        btn_p_day2.setProperty("class", "preset-pill")
        btn_p_day2.clicked.connect(self._load_day2_preset)

        preset_row.addWidget(btn_p_geo)
        preset_row.addWidget(btn_p_leo)
        preset_row.addWidget(btn_p_day2)
        hdr_layout.addLayout(preset_row)

        main_layout.addWidget(hdr_frame)

        # -------------------------------------------------------------
        # 2. Parameter Grid Sections
        # -------------------------------------------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)

        # Section A: Target Beacon & Kinematics
        sec_a = self._build_target_section()
        scroll_layout.addWidget(sec_a)

        # Section B: Virtual Camera & PAT Actuator
        sec_b = self._build_camera_section()
        scroll_layout.addWidget(sec_b)

        # Section C: Disturbance & Noise Subsystem
        sec_c = self._build_disturbance_section()
        scroll_layout.addWidget(sec_c)

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # -------------------------------------------------------------
        # 3. Action Buttons Footer
        # -------------------------------------------------------------
        footer_frame = QFrame()
        footer_frame.setProperty("class", "panel-card")
        footer_layout = QHBoxLayout(footer_frame)
        footer_layout.setContentsMargins(12, 10, 12, 10)
        footer_layout.setSpacing(10)

        self.btn_cancel = QPushButton("CANCEL")
        self.btn_cancel.setProperty("class", "secondary-btn")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_save_standby = QPushButton("APPLY & KEEP IN STANDBY")
        self.btn_save_standby.setProperty("class", "secondary-btn")
        self.btn_save_standby.clicked.connect(self._on_save_standby)

        self.btn_launch = QPushButton("🚀 APPLY & START SIMULATION")
        self.btn_launch.setProperty("class", "launch-btn")
        self.btn_launch.clicked.connect(self._on_save_and_launch)

        footer_layout.addWidget(self.btn_cancel)
        footer_layout.addStretch()
        footer_layout.addWidget(self.btn_save_standby)
        footer_layout.addWidget(self.btn_launch)

        main_layout.addWidget(footer_frame)

    # -------------------------------------------------------------
    # Section Builders
    # -------------------------------------------------------------
    def _build_target_section(self) -> QFrame:
        card = QFrame()
        card.setProperty("class", "panel-card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        hdr = QLabel("1. TARGET BEACON & MOTION TRAJECTORY")
        hdr.setProperty("class", "section-hdr")
        layout.addWidget(hdr)

        grid = QGridLayout()
        grid.setSpacing(10)

        # Row 0: Target Shape & Size
        grid.addWidget(QLabel("Target Shape:"), 0, 0)
        self.sp_shape = QComboBox()
        self.sp_shape.addItems(["circle", "square"])
        grid.addWidget(self.sp_shape, 0, 1)

        grid.addWidget(QLabel("Spot Size:"), 0, 2)
        size_box = QHBoxLayout()
        self.sp_size = QSpinBox()
        self.sp_size.setRange(5, 20)
        self.sp_size.setValue(10)
        size_box.addWidget(self.sp_size)
        size_box.addWidget(QLabel("px"))
        grid.addLayout(size_box, 0, 3)

        # Row 1: Intensity & Motion Pattern
        grid.addWidget(QLabel("Color Intensity:"), 1, 0)
        int_box = QHBoxLayout()
        self.sp_intensity = QSpinBox()
        self.sp_intensity.setRange(50, 255)
        self.sp_intensity.setValue(255)
        int_box.addWidget(self.sp_intensity)
        int_box.addWidget(QLabel("(0-255)"))
        grid.addLayout(int_box, 1, 1)

        grid.addWidget(QLabel("Motion Pattern:"), 1, 2)
        self.sp_pattern = QComboBox()
        self.sp_pattern.addItems([
            "straight_line",
            "circular",
            "figure_of_8",
            "random"
        ])
        grid.addWidget(self.sp_pattern, 1, 3)

        # Row 2: Speed & Linear Angle
        grid.addWidget(QLabel("Target Speed:"), 2, 0)
        spd_box = QHBoxLayout()
        self.sp_speed = QDoubleSpinBox()
        self.sp_speed.setRange(5.0, 150.0)
        self.sp_speed.setValue(40.0)
        self.sp_speed.setSingleStep(5.0)
        spd_box.addWidget(self.sp_speed)
        spd_box.addWidget(QLabel("px/s"))
        grid.addLayout(spd_box, 2, 1)

        grid.addWidget(QLabel("Linear Angle:"), 2, 2)
        ang_box = QHBoxLayout()
        self.sp_angle = QDoubleSpinBox()
        self.sp_angle.setRange(0.0, 360.0)
        self.sp_angle.setValue(35.0)
        self.sp_angle.setSingleStep(5.0)
        ang_box.addWidget(self.sp_angle)
        ang_box.addWidget(QLabel("deg"))
        grid.addLayout(ang_box, 2, 3)

        layout.addLayout(grid)
        return card

    def _build_camera_section(self) -> QFrame:
        card = QFrame()
        card.setProperty("class", "panel-card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        hdr = QLabel("2. VIRTUAL CAMERA, FIELD-OF-VIEW & PAT ACTUATOR")
        hdr.setProperty("class", "section-hdr")
        layout.addWidget(hdr)

        grid = QGridLayout()
        grid.setSpacing(10)

        # Row 0: Resolution & FOV
        grid.addWidget(QLabel("Sensor Resolution:"), 0, 0)
        lbl_res = QLabel("640 × 480 px (Sony IMX Crop)")
        lbl_res.setStyleSheet("color: #00e676; font-family: Consolas, monospace; font-size: 10px; font-weight: bold;")
        grid.addWidget(lbl_res, 0, 1)

        grid.addWidget(QLabel("Horizontal FOV:"), 0, 2)
        fov_h_box = QHBoxLayout()
        self.sp_fov_h = QDoubleSpinBox()
        self.sp_fov_h.setRange(1.0, 15.0)
        self.sp_fov_h.setValue(4.0)
        fov_h_box.addWidget(self.sp_fov_h)
        fov_h_box.addWidget(QLabel("deg"))
        grid.addLayout(fov_h_box, 0, 3)

        # Row 1: Max Slew Speed & Control Rate
        grid.addWidget(QLabel("Max Gimbal Speed:"), 1, 0)
        slew_box = QHBoxLayout()
        self.sp_slew = QDoubleSpinBox()
        self.sp_slew.setRange(1.0, 20.0)
        self.sp_slew.setValue(5.0)
        slew_box.addWidget(self.sp_slew)
        slew_box.addWidget(QLabel("deg/s"))
        grid.addLayout(slew_box, 1, 1)

        grid.addWidget(QLabel("Control Loop Rate:"), 1, 2)
        rate_box = QHBoxLayout()
        self.sp_rate = QDoubleSpinBox()
        self.sp_rate.setRange(10.0, 60.0)
        self.sp_rate.setValue(30.0)
        rate_box.addWidget(self.sp_rate)
        rate_box.addWidget(QLabel("Hz"))
        grid.addLayout(rate_box, 1, 3)

        layout.addLayout(grid)
        return card

    def _build_disturbance_section(self) -> QFrame:
        card = QFrame()
        card.setProperty("class", "panel-card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        hdr = QLabel("3. ENVIRONMENT, ATMOSPHERE & NOISE DISTURBANCES")
        hdr.setProperty("class", "section-hdr")
        layout.addWidget(hdr)

        grid = QGridLayout()
        grid.setSpacing(10)

        # Atmosphere condition
        grid.addWidget(QLabel("Atmosphere Condition:"), 0, 0)
        self.sp_atmo = QComboBox()
        self.sp_atmo.addItems(["Clear", "Rain", "Haze", "Fog", "Low Light"])
        grid.addWidget(self.sp_atmo, 0, 1)

        grid.addWidget(QLabel("Atmosphere Severity:"), 0, 2)
        sev_box = QHBoxLayout()
        self.sp_atmo_sev = QDoubleSpinBox()
        self.sp_atmo_sev.setRange(0.1, 1.0)
        self.sp_atmo_sev.setSingleStep(0.1)
        self.sp_atmo_sev.setValue(0.5)
        sev_box.addWidget(self.sp_atmo_sev)
        sev_box.addWidget(QLabel("(0.1-1.0)"))
        grid.addLayout(sev_box, 0, 3)

        # Noise checkboxes & parameter controls
        self.chk_gaussian = QCheckBox("Gaussian Sensor Noise")
        grid.addWidget(self.chk_gaussian, 1, 0)
        g_box = QHBoxLayout()
        self.sp_g_std = QDoubleSpinBox()
        self.sp_g_std.setRange(5.0, 50.0)
        self.sp_g_std.setValue(15.0)
        g_box.addWidget(QLabel("Std Dev:"))
        g_box.addWidget(self.sp_g_std)
        grid.addLayout(g_box, 1, 1)

        self.chk_sp = QCheckBox("Salt & Pepper Noise")
        grid.addWidget(self.chk_sp, 1, 2)
        sp_box = QHBoxLayout()
        self.sp_density = QDoubleSpinBox()
        self.sp_density.setRange(0.01, 0.30)
        self.sp_density.setSingleStep(0.02)
        self.sp_density.setValue(0.05)
        sp_box.addWidget(QLabel("Density:"))
        sp_box.addWidget(self.sp_density)
        grid.addLayout(sp_box, 1, 3)

        self.chk_jitter = QCheckBox("Camera Jitter (Vibration)")
        grid.addWidget(self.chk_jitter, 2, 0)
        j_box = QHBoxLayout()
        self.sp_jitter_max = QDoubleSpinBox()
        self.sp_jitter_max.setRange(1.0, 25.0)
        self.sp_jitter_max.setValue(10.0)
        j_box.addWidget(QLabel("Max px:"))
        j_box.addWidget(self.sp_jitter_max)
        grid.addLayout(j_box, 2, 1)

        self.chk_platform = QCheckBox("Platform Drift (Linear)")
        grid.addWidget(self.chk_platform, 2, 2)
        p_box = QHBoxLayout()
        self.sp_platform_max = QDoubleSpinBox()
        self.sp_platform_max.setRange(1.0, 25.0)
        self.sp_platform_max.setValue(10.0)
        p_box.addWidget(QLabel("Max px:"))
        p_box.addWidget(self.sp_platform_max)
        grid.addLayout(p_box, 2, 3)

        layout.addLayout(grid)
        return card

    # -------------------------------------------------------------
    # Presets
    # -------------------------------------------------------------
    def _load_geo_preset(self):
        self.sp_pattern.setCurrentText("straight_line")
        self.sp_speed.setValue(25.0)
        self.sp_atmo.setCurrentText("Clear")
        self.chk_gaussian.setChecked(False)
        self.chk_sp.setChecked(False)
        self.chk_jitter.setChecked(False)
        self.chk_platform.setChecked(False)

    def _load_leo_preset(self):
        self.sp_pattern.setCurrentText("circular")
        self.sp_speed.setValue(80.0)
        self.sp_atmo.setCurrentText("Clear")
        self.chk_gaussian.setChecked(True)
        self.chk_sp.setChecked(False)
        self.chk_jitter.setChecked(True)
        self.chk_platform.setChecked(False)

    def _load_day2_preset(self):
        self.sp_pattern.setCurrentText("straight_line")
        self.sp_speed.setValue(36.0)
        self.sp_atmo.setCurrentText("Rain")
        self.chk_gaussian.setChecked(False)
        self.chk_sp.setChecked(True)
        self.chk_jitter.setChecked(False)
        self.chk_platform.setChecked(False)

    # -------------------------------------------------------------
    # Configuration Load / Save
    # -------------------------------------------------------------
    def _load_from_config(self):
        self.sp_shape.setCurrentText(self.config.target_shape)
        self.sp_size.setValue(self.config.target_size_px)
        self.sp_intensity.setValue(self.config.target_color_intensity)
        self.sp_pattern.setCurrentText(self.config.motion_pattern)
        self.sp_speed.setValue(self.config.motion_speed_px_s)
        self.sp_fov_h.setValue(self.config.camera_fov_h_deg)
        self.sp_slew.setValue(self.config.max_pan_speed_deg_s)
        self.sp_rate.setValue(self.config.control_update_rate_hz)

        if self.dist_engine:
            self.sp_atmo.setCurrentText(self.dist_engine.atmospheric_condition)
            self.sp_atmo_sev.setValue(self.dist_engine.atmosphere_severity)
            self.chk_gaussian.setChecked(self.dist_engine.enable_gaussian)
            self.sp_g_std.setValue(self.dist_engine.gaussian_std_dev)
            self.chk_sp.setChecked(self.dist_engine.enable_salt_pepper)
            self.sp_density.setValue(self.dist_engine.sp_density)
            self.chk_jitter.setChecked(self.dist_engine.enable_jitter)
            self.sp_jitter_max.setValue(self.dist_engine.jitter_max_px)
            self.chk_platform.setChecked(self.dist_engine.enable_platform_motion)
            self.sp_platform_max.setValue(self.dist_engine.platform_max_px)

    def apply_to_config_and_runner(self):
        self.config.target_shape = self.sp_shape.currentText()
        self.config.target_size_px = self.sp_size.value()
        self.config.target_color_intensity = self.sp_intensity.value()
        self.config.motion_pattern = self.sp_pattern.currentText()
        self.config.motion_speed_px_s = self.sp_speed.value()
        self.config.camera_fov_h_deg = self.sp_fov_h.value()
        self.config.max_pan_speed_deg_s = self.sp_slew.value()
        self.config.max_tilt_speed_deg_s = self.sp_slew.value()
        self.config.control_update_rate_hz = self.sp_rate.value()

        if self.dist_engine:
            self.dist_engine.atmospheric_condition = self.sp_atmo.currentText()
            self.dist_engine.atmosphere_severity = self.sp_atmo_sev.value()
            self.dist_engine.enable_gaussian = self.chk_gaussian.isChecked()
            self.dist_engine.gaussian_std_dev = self.sp_g_std.value()
            self.dist_engine.enable_salt_pepper = self.chk_sp.isChecked()
            self.dist_engine.sp_density = self.sp_density.value()
            self.dist_engine.enable_jitter = self.chk_jitter.isChecked()
            self.dist_engine.jitter_max_px = self.sp_jitter_max.value()
            self.dist_engine.enable_platform_motion = self.chk_platform.isChecked()
            self.dist_engine.platform_max_px = self.sp_platform_max.value()

    def _on_save_standby(self):
        self.apply_to_config_and_runner()
        self.start_immediately = False
        self.accept()

    def _on_save_and_launch(self):
        self.apply_to_config_and_runner()
        self.start_immediately = True
        self.accept()
