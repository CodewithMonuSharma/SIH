"""
OptiTrack — FSOC Coarse Alignment Mission Control
Custom PySide6 High-Precision Aerospace Telemetry & Live Viewport Widgets
"""

import math
import time
from typing import Tuple, List, Optional
import numpy as np
import cv2

from PySide6.QtCore import Qt, QRectF, QPointF, QSize, QTimer
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QImage, QPixmap,
    QPainterPath, QLinearGradient, QRadialGradient
)
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QPushButton,
    QSizePolicy, QSlider
)


class OptiTrackSceneOverviewWidget(QWidget):
    """
    Twin Live-View: Scene Overview Canvas (2000x2000 px World Space representation).
    Matches the OptiTrack space-sensor HUD aesthetic:
      - Hex/Cartesian fine grid & polar range circles
      - Comet-tail dotted gradient trajectory curve
      - Target beacon with pulsing ripple & coordinate tag [TGT_01: X, Y]
      - Camera Gimbal FOV bounding box [640x480] with corner brackets and live AZ/EL readouts
      - Platform drift & range telemetry overlays
    """
    def __init__(self, world_width: int = 2000, world_height: int = 2000, parent=None):
        super().__init__(parent)
        self.world_w = world_width
        self.world_h = world_height
        self.target_pos = (1624.0, 580.0)
        self.gimbal_center = (1000.0, 1000.0)
        self.cam_fov_w = 640
        self.cam_fov_h = 480
        self.trail_history: List[Tuple[float, float]] = []
        self.max_trail_points = 60
        self.pan_deg = 14.2
        self.tilt_deg = 28.9
        self.platform_drift = 0.18
        self.range_km = 42.84
        self.is_aligned = True

        self.setMinimumSize(460, 340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def update_telemetry(self, target_pos: Tuple[float, float], gimbal_center: Tuple[float, float],
                         pan_deg: float, tilt_deg: float, drift: float = 0.18, is_aligned: bool = True):
        self.target_pos = target_pos
        self.gimbal_center = gimbal_center
        self.pan_deg = pan_deg
        self.tilt_deg = tilt_deg
        self.platform_drift = drift
        self.is_aligned = is_aligned

        if not self.trail_history or math.hypot(target_pos[0] - self.trail_history[-1][0], target_pos[1] - self.trail_history[-1][1]) > 4.0:
            self.trail_history.append(target_pos)
            if len(self.trail_history) > self.max_trail_points:
                self.trail_history.pop(0)

        self.update()

    def reset_trail(self):
        self.trail_history.clear()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # 1. Authentic Dark Space Viewport Background
        painter.fillRect(0, 0, w, h, QColor("#06090E"))

        scale_x = w / float(self.world_w)
        scale_y = h / float(self.world_h)

        def to_screen(wx, wy):
            return wx * scale_x, wy * scale_y

        # 2. Cartesian 100px / 200px World Grid
        pen_grid = QPen(QColor(31, 230, 196, 18), 1, Qt.SolidLine)
        painter.setPen(pen_grid)
        for x in range(0, self.world_w + 1, 200):
            sx, _ = to_screen(x, 0)
            painter.drawLine(int(sx), 0, int(sx), h)
        for y in range(0, self.world_h + 1, 200):
            _, sy = to_screen(0, y)
            painter.drawLine(0, int(sy), w, int(sy))

        # 3. Origin Crosshairs & Polar Concentric Rings
        pen_origin = QPen(QColor(31, 230, 196, 45), 1, Qt.DashLine)
        painter.setPen(pen_origin)
        cx, cy = w / 2.0, h / 2.0
        painter.drawLine(int(cx), 0, int(cx), h)
        painter.drawLine(0, int(cy), w, int(cy))

        pen_ring = QPen(QColor(31, 230, 196, 25), 1, Qt.SolidLine)
        painter.setPen(pen_ring)
        painter.setBrush(Qt.NoBrush)
        for r_ratio in [0.25, 0.5, 0.8]:
            r_px = min(w, h) * r_ratio * 0.5
            painter.drawEllipse(QPointF(cx, cy), r_px, r_px)

        # 4. Trajectory Comet-Tail Trail
        if len(self.trail_history) > 1:
            for i in range(len(self.trail_history) - 1):
                p1_s = to_screen(*self.trail_history[i])
                p2_s = to_screen(*self.trail_history[i + 1])
                alpha = int(255 * ((i + 1) / float(len(self.trail_history))) * 0.9)
                pen_comet = QPen(QColor(255, 207, 92, alpha), 2, Qt.CustomDashLine)
                pen_comet.setDashPattern([2, 4])
                painter.setPen(pen_comet)
                painter.drawLine(int(p1_s[0]), int(p1_s[1]), int(p2_s[0]), int(p2_s[1]))

        # 5. Target Beacon Representation
        tx_s, ty_s = to_screen(*self.target_pos)

        # Beacon outer halo
        painter.setPen(QPen(QColor(255, 207, 92, 100), 1))
        painter.setBrush(QBrush(QColor(255, 207, 92, 40)))
        painter.drawEllipse(QPointF(tx_s, ty_s), 10, 10)

        # Beacon core
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#FFCF5C")))
        painter.drawEllipse(QPointF(tx_s, ty_s), 4, 4)

        # Target coordinate tag
        painter.setFont(QFont("Consolas", 8, QFont.Bold))
        tag_text = f"TGT_01: [{int(self.target_pos[0])}, {int(self.target_pos[1])}]"
        tag_rect = QRectF(tx_s - 45, ty_s + 12, 90, 16)
        painter.fillRect(tag_rect, QColor(10, 14, 20, 220))
        painter.setPen(QPen(QColor(255, 207, 92, 160), 1))
        painter.drawRect(tag_rect)
        painter.setPen(QPen(QColor("#FFCF5C")))
        painter.drawText(tag_rect, Qt.AlignCenter, tag_text)

        # 6. Gimbal Camera FOV Rectangle (640x480 World Scale)
        fov_w_s = self.cam_fov_w * scale_x
        fov_h_s = self.cam_fov_h * scale_y
        gx_s, gy_s = to_screen(*self.gimbal_center)
        fov_x = gx_s - fov_w_s / 2.0
        fov_y = gy_s - fov_h_s / 2.0
        fov_rect = QRectF(fov_x, fov_y, fov_w_s, fov_h_s)

        # FOV Glass fill & border
        painter.fillRect(fov_rect, QColor(31, 230, 196, 12))
        pen_fov = QPen(QColor(31, 230, 196, 160), 1, Qt.SolidLine)
        painter.setPen(pen_fov)
        painter.drawRect(fov_rect)

        # Corner Brackets
        brk = 8
        painter.setPen(QPen(QColor("#1FE6C4"), 2, Qt.SolidLine))
        # Top-left
        painter.drawLine(int(fov_x), int(fov_y + brk), int(fov_x), int(fov_y))
        painter.drawLine(int(fov_x), int(fov_y), int(fov_x + brk), int(fov_y))
        # Top-right
        painter.drawLine(int(fov_x + fov_w_s - brk), int(fov_y), int(fov_x + fov_w_s), int(fov_y))
        painter.drawLine(int(fov_x + fov_w_s), int(fov_y), int(fov_x + fov_w_s), int(fov_y + brk))
        # Bottom-left
        painter.drawLine(int(fov_x), int(fov_y + fov_h_s - brk), int(fov_x), int(fov_y + fov_h_s))
        painter.drawLine(int(fov_x), int(fov_y + fov_h_s), int(fov_x + brk), int(fov_y + fov_h_s))
        # Bottom-right
        painter.drawLine(int(fov_x + fov_w_s - brk), int(fov_y + fov_h_s), int(fov_x + fov_w_s), int(fov_y + fov_h_s))
        painter.drawLine(int(fov_x + fov_w_s), int(fov_y + fov_h_s), int(fov_x + fov_w_s), int(fov_y + fov_h_s - brk))

        # FOV Labels
        painter.setFont(QFont("Consolas", 7, QFont.Bold))
        painter.setPen(QPen(QColor("#1FE6C4")))
        painter.drawText(int(fov_x + 5), int(fov_y + 12), "GIMBAL FOV [640×480]")
        az_el_text = f"AZ: {self.pan_deg:+.1f}° | EL: {self.tilt_deg:+.1f}°"
        painter.drawText(int(fov_x + fov_w_s - 110), int(fov_y + fov_h_s - 5), az_el_text)

        # 7. Top-Left Origin & Altitude HUD overlay
        overlay_rect = QRectF(10, 10, 140, 44)
        painter.fillRect(overlay_rect, QColor(10, 14, 20, 210))
        painter.setPen(QPen(QColor(31, 230, 196, 60), 1))
        painter.drawRect(overlay_rect)
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(16, 23, "ORIGIN: [1000, 1000]")
        painter.drawText(16, 36, "SAT ELEV: 520 km")
        painter.drawText(16, 49, "UAV ALT: 18.5 km")

        # 8. Bottom Telemetry Strip
        bot_rect = QRectF(10, h - 26, w - 20, 20)
        painter.fillRect(bot_rect, QColor(10, 14, 20, 230))
        painter.setPen(QPen(QColor(31, 230, 196, 70), 1))
        painter.drawRect(bot_rect)
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(18, h - 12, f"PLATFORM DRIFT: {self.platform_drift:.2f} px/s")
        painter.drawText(int(w * 0.35), h - 12, "EPHEMERIS: GPS-RTK SYNC")
        painter.setPen(QPen(QColor("#1FE6C4")))
        painter.drawText(int(w * 0.62), h - 12, f"RANGE: {self.range_km:.2f} km")

        # Status badge dot
        status_col = QColor("#3DDC84") if self.is_aligned else QColor("#FF6B5E")
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(status_col))
        painter.drawEllipse(QPointF(w - 110, h - 16), 3, 3)
        painter.setPen(QPen(status_col))
        painter.drawText(w - 100, h - 12, "COARSE ALIGNED" if self.is_aligned else "ACQUIRING...")


class OptiTrackCameraFeedWidget(QWidget):
    """
    Twin Live-View: Camera Sensor Viewport (640x480 Sensor Crop).
    Matches the OptiTrack precision lens chamber:
      - Renders real OpenCV processed sensor image (grayscale/RGB)
      - Technical HUD crosshairs, pitch/yaw ladders, concentric range circles
      - Kalman tracking reticle with corner brackets & sub-pixel coordinate deltas
      - Live exposure, gain, NIR beacon, servo response, and signal margin overlays
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_qimage: Optional[QImage] = None
        self.has_target = True
        self.centroid_pos = (320.0, 240.0)
        self.is_locked = True
        self.confidence = 99.4
        self.delta_x = 0.42
        self.delta_y = -0.19
        self.exposure_text = "EXPOSURE: 1/2000s"
        self.gain_text = "GAIN: 8.2 dB"
        self.beacon_text = "BEACON: 850nm NIR"
        self.servo_resp_ms = 12
        self.signal_margin_db = 16.8

        self.setMinimumSize(420, 340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def update_frame(self, frame_img: np.ndarray, obs_centroid: Optional[Tuple[float, float]] = None,
                     is_locked: bool = True, confidence: float = 99.4, delta_x: float = 0.0, delta_y: float = 0.0):
        # Convert numpy frame to QImage
        h, w = frame_img.shape[:2]
        if len(frame_img.shape) == 2:
            # Grayscale IR
            bytes_per_line = w
            self.current_qimage = QImage(frame_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()
        else:
            # BGR -> RGB
            rgb_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * w
            self.current_qimage = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

        self.has_target = obs_centroid is not None
        if obs_centroid:
            self.centroid_pos = obs_centroid
        self.is_locked = is_locked
        self.confidence = confidence
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # 1. Dark Lens Chamber Base
        painter.fillRect(0, 0, w, h, QColor("#04060A"))

        # 2. Draw Camera Frame (Scaled to fit viewport while maintaining 4:3 aspect ratio)
        if self.current_qimage and not self.current_qimage.isNull():
            scaled_img = self.current_qimage.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_x = (w - scaled_img.width()) / 2
            img_y = (h - scaled_img.height()) / 2
            painter.drawImage(int(img_x), int(img_y), scaled_img)
            view_w = scaled_img.width()
            view_h = scaled_img.height()
            off_x = img_x
            off_y = img_y
        else:
            view_w = w
            view_h = h
            off_x = 0
            off_y = 0

        # Scale helper from 640x480 native to rendered viewport
        sc_x = view_w / 640.0
        sc_y = view_h / 480.0
        cx = off_x + view_w / 2.0
        cy = off_y + view_h / 2.0

        # 3. Technical HUD Overlay: Concentric Circles & Crosshairs
        pen_hud_rings = QPen(QColor(31, 230, 196, 45), 1, Qt.DashLine)
        painter.setPen(pen_hud_rings)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), 60 * sc_x, 60 * sc_y)
        painter.drawEllipse(QPointF(cx, cy), 160 * sc_x, 160 * sc_y)
        painter.drawEllipse(QPointF(cx, cy), 220 * sc_x, 220 * sc_y)

        # Center Boresight Lines
        pen_boresight = QPen(QColor(31, 230, 196, 60), 1, Qt.CustomDashLine)
        pen_boresight.setDashPattern([2, 4])
        painter.setPen(pen_boresight)
        painter.drawLine(int(cx), int(off_y + 15), int(cx), int(off_y + view_h - 15))
        painter.drawLine(int(off_x + 15), int(cy), int(off_x + view_w - 15), int(cy))

        # Pitch & Yaw Ladder Bars
        pen_ladder = QPen(QColor("#1FE6C4"), 1.5, Qt.SolidLine)
        painter.setPen(pen_ladder)
        painter.drawLine(int(cx - 15), int(cy - 60 * sc_y), int(cx + 15), int(cy - 60 * sc_y))
        painter.drawLine(int(cx - 15), int(cy + 60 * sc_y), int(cx + 15), int(cy + 60 * sc_y))
        painter.drawLine(int(cx - 60 * sc_x), int(cy - 12), int(cx - 60 * sc_x), int(cy + 12))
        painter.drawLine(int(cx + 60 * sc_x), int(cy - 12), int(cx + 60 * sc_x), int(cy + 12))

        # Corner Lens Frame Markers
        c_len = 16
        painter.drawLine(int(off_x + 10), int(off_y + 10 + c_len), int(off_x + 10), int(off_y + 10))
        painter.drawLine(int(off_x + 10), int(off_y + 10), int(off_x + 10 + c_len), int(off_y + 10))
        painter.drawLine(int(off_x + view_w - 10 - c_len), int(off_y + 10), int(off_x + view_w - 10), int(off_y + 10))
        painter.drawLine(int(off_x + view_w - 10), int(off_y + 10), int(off_x + view_w - 10), int(off_y + 10 + c_len))
        painter.drawLine(int(off_x + 10), int(off_y + view_h - 10 - c_len), int(off_x + 10), int(off_y + view_h - 10))
        painter.drawLine(int(off_x + 10), int(off_y + view_h - 10), int(off_x + 10 + c_len), int(off_y + view_h - 10))
        painter.drawLine(int(off_x + view_w - 10 - c_len), int(off_y + view_h - 10), int(off_x + view_w - 10), int(off_y + view_h - 10))
        painter.drawLine(int(off_x + view_w - 10), int(off_y + view_h - 10), int(off_x + view_w - 10), int(off_y + view_h - 10 - c_len))

        # 4. Kalman Reticle Snap & Centroid Box
        if self.has_target:
            tx_render = off_x + self.centroid_pos[0] * sc_x
            ty_render = off_y + self.centroid_pos[1] * sc_y

            lock_col = QColor("#3DDC84") if self.is_locked else QColor("#FFCF5C")
            reticle_size = 38

            # Lock Reticle Square
            painter.setPen(QPen(lock_col, 2, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            r_box = QRectF(tx_render - reticle_size / 2.0, ty_render - reticle_size / 2.0, reticle_size, reticle_size)
            painter.drawRect(r_box)

            # Corner notches
            painter.fillRect(QRectF(tx_render - 3, ty_render - reticle_size / 2.0 - 2, 6, 2), lock_col)
            painter.fillRect(QRectF(tx_render - 3, ty_render + reticle_size / 2.0, 6, 2), lock_col)
            painter.fillRect(QRectF(tx_render - reticle_size / 2.0 - 2, ty_render - 3, 2, 6), lock_col)
            painter.fillRect(QRectF(tx_render + reticle_size / 2.0, ty_render - 3, 2, 6), lock_col)

            # Target Lock Tag
            tag_rect = QRectF(tx_render + 24, ty_render - 28, 126, 26)
            painter.fillRect(tag_rect, QColor(15, 21, 31, 230))
            painter.setPen(QPen(lock_col, 1))
            painter.drawRect(tag_rect)

            painter.setFont(QFont("Consolas", 7, QFont.Bold))
            painter.setPen(QPen(lock_col))
            painter.drawText(int(tx_render + 30), int(ty_render - 17), f"BEACON #01 | CONF: {self.confidence:.1f}%")
            painter.setPen(QPen(QColor("#F2F3F5")))
            painter.drawText(int(tx_render + 30), int(ty_render - 6), f"ΔX: {self.delta_x:+.2f}px · ΔY: {self.delta_y:+.2f}px")

        # 5. Top Optical Sensor Overlay
        top_rect = QRectF(off_x + 10, off_y + 10, 240, 18)
        painter.fillRect(top_rect, QColor(10, 14, 20, 210))
        painter.setPen(QPen(QColor(31, 230, 196, 60), 1))
        painter.drawRect(top_rect)
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QPen(QColor("#1FE6C4")))
        painter.drawText(int(off_x + 16), int(off_y + 22), f"{self.exposure_text} · {self.gain_text} · {self.beacon_text}")

        # 6. Bottom Optical Sensor Overlay
        bot_rect = QRectF(off_x + 10, off_y + view_h - 26, view_w - 20, 18)
        painter.fillRect(bot_rect, QColor(10, 14, 20, 220))
        painter.setPen(QPen(QColor(31, 230, 196, 60), 1))
        painter.drawRect(bot_rect)
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QPen(QColor("#8A8F9A")))
        painter.drawText(int(off_x + 16), int(off_y + view_h - 14), "ALGORITHM: ")
        painter.setPen(QPen(QColor("#1FE6C4")))
        painter.drawText(int(off_x + 72), int(off_y + view_h - 14), "KALMAN + CENTROID WEIGHTED")
        painter.setPen(QPen(QColor("#8A8F9A")))
        painter.drawText(int(off_x + view_w - 210), int(off_y + view_h - 14), f"SERVO: {self.servo_resp_ms}ms  |  MARGIN: +{self.signal_margin_db:.1f} dB")


class OptiTrackStatCard(QFrame):
    """
    Glass-Morphism Telemetry Stat Card for 6 metric parameters.
    """
    def __init__(self, title: str, initial_value: str = "0.00", unit: str = "px",
                 spec_text: str = "SPEC: < 1.5 px", status_text: str = "IN SPEC",
                 accent_color: str = "#0F9E85", parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #D8DCE3;
                border-radius: 8px;
            }
            QFrame:hover {
                border: 1px solid rgba(15, 158, 133, 0.6);
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        # Header: Title + Dot
        hdr = QHBoxLayout()
        self.lbl_title = QLabel(title.upper())
        self.lbl_title.setStyleSheet("font-size: 10px; font-family: Consolas, monospace; color: #6B7280; font-weight: bold; letter-spacing: 0.5px;")
        self.dot = QFrame()
        self.dot.setFixedSize(6, 6)
        self.dot.setStyleSheet(f"background-color: {accent_color}; border-radius: 3px;")
        hdr.addWidget(self.lbl_title)
        hdr.addStretch()
        hdr.addWidget(self.dot)
        layout.addLayout(hdr)

        # Value Row
        val_row = QHBoxLayout()
        val_row.setSpacing(4)
        self.lbl_val = QLabel(initial_value)
        self.lbl_val.setStyleSheet(f"font-size: 20px; font-weight: bold; font-family: Consolas, monospace; color: {accent_color};")
        self.lbl_unit = QLabel(unit)
        self.lbl_unit.setStyleSheet("font-size: 11px; font-family: Consolas, monospace; color: #6B7280; margin-top: 4px;")
        val_row.addWidget(self.lbl_val)
        val_row.addWidget(self.lbl_unit)
        val_row.addStretch()
        layout.addLayout(val_row)

        # Sub-status row
        sub_row = QHBoxLayout()
        self.lbl_spec = QLabel(spec_text)
        self.lbl_spec.setStyleSheet("font-size: 9px; font-family: Consolas, monospace; color: #6B7280;")
        self.lbl_status = QLabel(status_text)
        self.lbl_status.setStyleSheet("font-size: 9px; font-family: Consolas, monospace; font-weight: bold; color: #1FAE64;")
        sub_row.addWidget(self.lbl_spec)
        sub_row.addStretch()
        sub_row.addWidget(self.lbl_status)
        layout.addLayout(sub_row)

    def set_value(self, val_str: str, status_str: Optional[str] = None, is_ok: bool = True):
        self.lbl_val.setText(val_str)
        if status_str is not None:
            self.lbl_status.setText(status_str)
            self.lbl_status.setStyleSheet("font-size: 9px; font-family: Consolas, monospace; font-weight: bold; color: #1FAE64;" if is_ok else "font-size: 9px; font-family: Consolas, monospace; font-weight: bold; color: #DC2626;")


class TopMetricPill(QFrame):
    """Compact Top Metric Pill Widget."""
    def __init__(self, title: str, value: str = "0.0", color_hex: str = "#0F9E85", parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #F7F8FA;
                border: 1px solid #D8DCE3;
                border-radius: 4px;
            }
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(4)
        self.lbl_t = QLabel(title)
        self.lbl_t.setStyleSheet("font-size: 8px; font-family: Consolas, monospace; color: #6B7280; font-weight: bold;")
        self.lbl_v = QLabel(value)
        self.lbl_v.setStyleSheet(f"font-size: 9px; font-family: Consolas, monospace; font-weight: bold; color: {color_hex};")
        layout.addWidget(self.lbl_t)
        layout.addWidget(self.lbl_v)

    def set_value(self, val_str: str, color_hex: Optional[str] = None):
        self.lbl_v.setText(val_str)
        if color_hex:
            self.lbl_v.setStyleSheet(f"font-size: 9px; font-family: Consolas, monospace; font-weight: bold; color: {color_hex};")


# Backward compatibility aliases
VirtualArenaRadarMap = OptiTrackSceneOverviewWidget
WorldMapWidget = OptiTrackSceneOverviewWidget
SleekMetricCard = OptiTrackStatCard
MetricCard = OptiTrackStatCard

