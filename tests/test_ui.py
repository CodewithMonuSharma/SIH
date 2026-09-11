import os
import pytest

def test_ui_instantiation():
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from ui.main_window import CoarsePATMainWindow

    win = CoarsePATMainWindow(auto_start=False)
    assert win is not None
    assert win.is_running is False

    # Test run/pause toggling
    win._on_run()
    assert win.is_running is True

    win._on_pause()
    assert win.is_running is False

    # Test motion pattern switching
    win._set_motion_pattern("circular")
    win._set_motion_pattern("straight")

    # Test theme toggle
    win._toggle_theme()
    assert win.current_theme == "dark"
    win._toggle_theme()
    assert win.current_theme == "light"

    # Test reset
    win._on_reset()
    assert win.is_running is False

def test_config_dialog():
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from config.schemas import SimConfig
    from disturbance.engine import DisturbanceEngine
    from ui.config_dialog import SimulationConfigDialog

    cfg = SimConfig()
    dist = DisturbanceEngine()
    dlg = SimulationConfigDialog(cfg, dist)

    # Test loading preset
    dlg._load_day2_preset()
    assert dlg.sp_pattern.currentText() == "straight_line"
    assert dlg.sp_atmo.currentText() == "Rain"
    assert dlg.chk_sp.isChecked() is True

    # Test user entering custom values
    dlg.sp_shape.setCurrentText("circle")
    dlg.sp_size.setValue(14)
    dlg.sp_speed.setValue(55.0)
    dlg.sp_slew.setValue(8.0)

    dlg.apply_to_config_and_runner()
    assert cfg.target_shape == "circle"
    assert cfg.target_size_px == 14
    assert cfg.motion_speed_px_s == 55.0
    assert cfg.max_pan_speed_deg_s == 8.0
