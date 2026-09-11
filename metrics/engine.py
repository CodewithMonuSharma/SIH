import math
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from tracking.state import TrackingState
from tracking.error import TrackingErrorResult

@dataclass
class PerformanceSummaryReport:
    """
    FR-23 Performance Log Summary Deliverable.
    Aggregates run-level performance statistics according to official PS 26169 Section 8 formulas.
    """
    total_frames: int
    simulation_duration_sec: float
    fps: float
    acquisition_time_sec: Optional[float]
    reacquisition_time_sec: Optional[float]
    mean_tracking_error_px: float
    max_tracking_error_px: float
    rmse_px: float
    target_loss_percent: float
    lock_retention_rate_percent: float
    avg_processing_latency_ms: float

class PerformanceMeasurementEngine:
    """
    FR-23 Performance Measurement Engine.
    Aggregates per-frame tracking observations, tracking state transitions,
    and frame processing latencies into real-time & end-of-run summary metrics.
    """
    def __init__(self):
        self.frame_records: List[Dict[str, Any]] = []
        self.errors: List[float] = []
        self.latencies_ms: List[float] = []

        self.t_start = time.time()
        self.t_acquisition: Optional[float] = None
        self.t_reacquisition: Optional[float] = None

        self.locked_frames = 0
        self.lost_frames = 0
        self.total_processed_frames = 0

    def record_frame(self, frame_idx: int, timestamp: float, state: TrackingState,
                     error_res: Optional[TrackingErrorResult], processing_latency_ms: float,
                     meta: Optional[Dict[str, Any]] = None):
        self.total_processed_frames += 1
        self.latencies_ms.append(processing_latency_ms)

        err_mag = error_res.error_magnitude if error_res else None
        if err_mag is not None:
            self.errors.append(err_mag)

        if state == TrackingState.LOCKED:
            self.locked_frames += 1
        elif state == TrackingState.TARGET_LOST or state == TrackingState.REACQUISITION:
            self.lost_frames += 1

        rec = {
            "frame_id": frame_idx,
            "timestamp": timestamp,
            "tracking_state": state.value,
            "error_px": err_mag,
            "latency_ms": processing_latency_ms
        }
        if meta:
            rec.update(meta)
        self.frame_records.append(rec)

    def generate_summary(self, acquisition_time_sec: Optional[float] = None,
                         reacquisition_time_sec: Optional[float] = None) -> PerformanceSummaryReport:
        elapsed = time.time() - self.t_start
        fps = self.total_processed_frames / elapsed if elapsed > 0 else 0.0

        mean_err = float(sum(self.errors) / len(self.errors)) if self.errors else 0.0
        max_err = float(max(self.errors)) if self.errors else 0.0
        rmse = float(math.sqrt(sum(e**2 for e in self.errors) / len(self.errors))) if self.errors else 0.0

        target_loss_pct = (self.lost_frames / self.total_processed_frames * 100.0) if self.total_processed_frames > 0 else 0.0
        lock_retention_pct = (self.locked_frames / self.total_processed_frames * 100.0) if self.total_processed_frames > 0 else 0.0
        avg_lat = sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

        return PerformanceSummaryReport(
            total_frames=self.total_processed_frames,
            simulation_duration_sec=round(elapsed, 2),
            fps=round(fps, 1),
            acquisition_time_sec=round(acquisition_time_sec, 3) if acquisition_time_sec else None,
            reacquisition_time_sec=round(reacquisition_time_sec, 3) if reacquisition_time_sec else None,
            mean_tracking_error_px=round(mean_err, 2),
            max_tracking_error_px=round(max_err, 2),
            rmse_px=round(rmse, 2),
            target_loss_percent=round(target_loss_pct, 2),
            lock_retention_rate_percent=round(lock_retention_pct, 2),
            avg_processing_latency_ms=round(avg_lat, 2)
        )
