import math
from typing import Optional, Tuple
from frame.schema import Frame
from detection.base import BaseDetector
from tracking.observation import CentroidObservation

class ReacquisitionSearchStrategy:
    """
    FR-22 Re-acquisition Subsystem (Active Camera Search).

    ROOT CAUSE OF ORIGINAL BUG:
    When the target leaves the camera viewport, the camera was frozen (pan_cmd=0, tilt_cmd=0)
    and only ran the detector on the same still frame — which obviously never found the target.

    FIX: On each REACQUISITION tick, this class returns a (pan_cmd, tilt_cmd) sweep velocity
    command via get_sweep_command(), which the SimulationRunner applies to the camera actuator
    so it actively sweeps to find the target again.

    Search Strategy: Outward Lissajous Spiral Sweep
    - Uses a time-parameterized spiral pattern centered on the last known target position.
    - On each re-acquisition frame, the camera moves outward along this spiral.
    - As soon as the detector finds the target inside the new viewport, re-lock is declared.
    """

    def __init__(self, detector: BaseDetector,
                 sweep_speed_deg_s: float = 4.0,
                 max_sweep_radius_deg: float = 2.5,
                 angular_freq: float = 1.2):
        self.detector = detector
        self.sweep_speed = sweep_speed_deg_s
        self.max_radius = max_sweep_radius_deg
        self.angular_freq = angular_freq

        # Spiral sweep state
        self._sweep_t: float = 0.0          # Sweep timer (resets on each new loss event)
        self._center_pan: float = 0.0       # Pan angle when target was last seen
        self._center_tilt: float = 0.0      # Tilt angle when target was last seen

    def reset_sweep(self, last_known_pan: float, last_known_tilt: float):
        """
        Called when the camera enters TARGET_LOST.
        Records the last known camera pointing direction so the spiral is centered there.
        """
        self._sweep_t = 0.0
        self._center_pan = last_known_pan
        self._center_tilt = last_known_tilt

    def get_sweep_command(self, dt: float) -> Tuple[float, float]:
        """
        Returns (pan_cmd_deg_s, tilt_cmd_deg_s) sweep velocity command for one tick.

        Spiral Search Pattern:
          r(t) = max_radius * (1 - exp(-k*t))   (radius grows outward from center)
          pan_offset  = r(t) * cos(angular_freq * t)
          tilt_offset = r(t) * sin(angular_freq * t)

        The camera velocity is the derivative of offset w.r.t. time, approximated as
        finite-difference: (offset(t+dt) - offset(t)) / dt for simplicity.
        """
        self._sweep_t += dt
        t = self._sweep_t
        k = 0.5  # Expansion rate constant

        # Outward growing radius
        r = self.max_radius * (1.0 - math.exp(-k * t))
        r_next = self.max_radius * (1.0 - math.exp(-k * (t + dt)))

        pan_offset = r * math.cos(self.angular_freq * t)
        tilt_offset = r * math.sin(self.angular_freq * t)

        pan_offset_next = r_next * math.cos(self.angular_freq * (t + dt))
        tilt_offset_next = r_next * math.sin(self.angular_freq * (t + dt))

        # Finite-difference velocity command (deg/s)
        pan_cmd = (pan_offset_next - pan_offset) / dt
        tilt_cmd = (tilt_offset_next - tilt_offset) / dt

        # Clamp to physical actuator speed limit
        pan_cmd = max(-self.sweep_speed, min(self.sweep_speed, pan_cmd))
        tilt_cmd = max(-self.sweep_speed, min(self.sweep_speed, tilt_cmd))

        return pan_cmd, tilt_cmd

    def search(self, frame: Frame) -> Optional[CentroidObservation]:
        """
        Runs the detector on whatever the camera currently sees after the sweep move.
        The camera is moved BEFORE this is called (by SimulationRunner.step()).
        """
        return self.detector.detect(frame)
