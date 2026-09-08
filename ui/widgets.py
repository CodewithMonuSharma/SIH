from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QImage, QPixmap, QFont
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame
import numpy as np

class WorldMapWidget(QWidget):
    """
    Custom 2D Virtual World Mini-Map View (2000x2000 Canvas Overview).
    Visualizes the entire world scene, target trajectory position, and
    the moving camera FOV crop rectangle in real time.
    """
    def __init__(self, world_width: int = 2000, world_height: int = 2000, parent=None):
        super().__init__(parent)
        self.world_w = world_width
        self.world_h = world_height
        self.target_pos = (1000.0, 1000.0)
        self.cam_fov_rect = (680, 760, 640, 480)  # x1, y1, width, height
        self.setMinimumSize(320, 320)
        self.setStyleSheet("background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px;")

    def update_state(self, target_pos: tuple, cam_fov_rect: tuple):
        self.target_pos = target_pos
        self.cam_fov_rect = cam_fov_rect
        self.update()  # Trigger paintEvent

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Fill background
        painter.fillRect(0, 0, w, h, QColor("#0d1117"))

        # Draw grid
        pen_grid = QPen(QColor("#21262d"), 1, Qt.DashLine)
        painter.setPen(pen_grid)
        grid_cols, grid_rows = 5, 5
        for i in range(1, grid_cols):
            x = i * (w / grid_cols)
            painter.drawLine(int(x), 0, int(x), h)
        for j in range(1, grid_rows):
            y = j * (h / grid_rows)
            painter.drawLine(0, int(y), w, int(y))

        scale_x = w / self.world_w
        scale_y = h / self.world_h

        # Draw Camera FOV Window
        fx, fy, fw, fh = self.cam_fov_rect
        fov_x = fx * scale_x
        fov_y = fy * scale_y
        fov_w = fw * scale_x
        fov_h = fh * scale_y

        pen_fov = QPen(QColor("#58a6ff"), 2, Qt.SolidLine)
        painter.setPen(pen_fov)
        painter.setBrush(QBrush(QColor(88, 166, 255, 35)))
        painter.drawRect(QRectF(fov_x, fov_y, fov_w, fov_h))

        # Draw FOV Boresight Center
        cam_cx = fov_x + fov_w / 2.0
        cam_cy = fov_y + fov_h / 2.0
        pen_cross = QPen(QColor("#58a6ff"), 1, Qt.SolidLine)
        painter.setPen(pen_cross)
        painter.drawLine(int(cam_cx - 6), int(cam_cy), int(cam_cx + 6), int(cam_cy))
        painter.drawLine(int(cam_cx), int(cam_cy - 6), int(cam_cx), int(cam_cy + 6))

        # Draw Target Beacon Spot
        tx, ty = self.target_pos
        t_mapped_x = tx * scale_x
        t_mapped_y = ty * scale_y

        painter.setPen(QPen(QColor("#f85149"), 1))
        painter.setBrush(QBrush(QColor("#ff7b72")))
        painter.drawEllipse(QPointF(t_mapped_x, t_mapped_y), 5, 5)

        # Labels
        painter.setPen(QPen(QColor("#8b949e")))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(10, 20, "VIRTUAL WORLD (2000x2000)")

class MetricCard(QFrame):
    """
    Stylized Telemetry Data Card Widget.
    Displays metric label, live value, unit, and status highlight.
    """
    def __init__(self, title: str, initial_value: str = "0.0", unit: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(2)

        self.title_label = QLabel(title.upper())
        self.title_label.setStyleSheet("color: #8b949e; font-size: 10px; font-weight: bold;")
        
        self.value_label = QLabel(initial_value)
        self.value_label.setStyleSheet("color: #58a6ff; font-size: 18px; font-weight: bold; font-family: Consolas, monospace;")

        self.unit_label = QLabel(unit)
        self.unit_label.setStyleSheet("color: #8b949e; font-size: 10px;")

        layout.addWidget(self.title_label)
        
        val_row = QHBoxLayout()
        val_row.setContentsMargins(0, 0, 0, 0)
        val_row.addWidget(self.value_label)
        val_row.addWidget(self.unit_label)
        val_row.addStretch()
        
        layout.addLayout(val_row)

    def set_value(self, value_str: str, color_hex: str = "#58a6ff"):
        self.value_label.setText(value_str)
        self.value_label.setStyleSheet(f"color: {color_hex}; font-size: 18px; font-weight: bold; font-family: Consolas, monospace;")
