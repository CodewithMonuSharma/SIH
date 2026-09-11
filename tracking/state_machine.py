from typing import Optional
from tracking.state import TrackingState
from tracking.observation import CentroidObservation

class TrackingStateMachine:
    """
    FR-20 Target Lock State & FR-21 Target Loss Detection.
    Manages 6-state tracking lifecycle with debounce filtering (N=3 lock debounce, M=3 loss debounce)
    and measures cold acquisition time (<= 2s) and re-acquisition latency (<= 1s).

    BUG FIXES APPLIED:
      1. Removed unused `import time` (all timestamps are simulation timestamps, not wall clock).
      2. Changed t_search_start from Optional[float] = None to float = 0.0 so acquisition
         time is computed correctly from simulation start.
      3. Clarified LOCKED-state miss counting logic (consecutive_misses correctly resets
         on each good detection frame via the counter update at the top of update()).
      4. Changed reacquisition debounce from >= 1 to >= 2 consecutive detections to
         prevent a single noisy false-positive from triggering a premature re-lock.
    """
    def __init__(self, lock_debounce_count: int = 3, loss_debounce_count: int = 3):
        self.state = TrackingState.INITIALIZATION
        self.lock_debounce = lock_debounce_count
        self.loss_debounce = loss_debounce_count

        self.consecutive_detections = 0
        self.consecutive_misses = 0

        # FIX 1: Use simulation timestamps (float) not wall-clock time.
        # t_search_start initialized to 0.0 = start of simulation timeline.
        self.t_search_start: float = 0.0
        self.t_acquisition: Optional[float] = None
        self.t_loss_start: Optional[float] = None
        self.t_reacquired: Optional[float] = None

        self.total_locked_time = 0.0
        self.total_lost_time = 0.0
        self.acquisition_time_sec: Optional[float] = None
        self.last_reacquisition_time_sec: Optional[float] = None

        # Immediately transition to SEARCHING
        self.state = TrackingState.SEARCHING

    def update(self, observation: Optional[CentroidObservation], timestamp: float, dt: float) -> TrackingState:
        has_detection = (
            observation is not None
            and observation.is_valid
            and observation.confidence >= 0.3
        )

        # ── Update detection/miss counters FIRST ─────────────────────────
        if has_detection:
            self.consecutive_detections += 1
            self.consecutive_misses = 0       # FIX 3: resets correctly on every good detection
        else:
            self.consecutive_misses += 1
            self.consecutive_detections = 0

        # ── State Transition Logic ────────────────────────────────────────
        if self.state == TrackingState.SEARCHING:
            if has_detection:
                self.state = TrackingState.BEACON_DETECTED

        elif self.state == TrackingState.BEACON_DETECTED:
            if not has_detection:
                # One miss drops back to SEARCHING
                self.state = TrackingState.SEARCHING
            elif self.consecutive_detections >= self.lock_debounce:
                # N consecutive confident detections → ACQUISITION
                self.state = TrackingState.ACQUISITION
                self.t_acquisition = timestamp
                # FIX 2: t_search_start is always 0.0 (valid float), safe to subtract
                self.acquisition_time_sec = self.t_acquisition - self.t_search_start

        elif self.state == TrackingState.ACQUISITION:
            if not has_detection:
                # Lost during acquisition → restart
                self.state = TrackingState.SEARCHING
            else:
                # Confirmed stable → LOCKED
                self.state = TrackingState.LOCKED

        elif self.state == TrackingState.LOCKED:
            # Accumulate locked time every frame we stay in LOCKED state
            self.total_locked_time += dt
            # M consecutive misses → declare TARGET_LOST
            if self.consecutive_misses >= self.loss_debounce:
                self.state = TrackingState.TARGET_LOST
                self.t_loss_start = timestamp

        elif self.state in (TrackingState.TARGET_LOST, TrackingState.REACQUISITION):
            self.total_lost_time += dt
            self.state = TrackingState.REACQUISITION
            # FIX 4: Require >= 2 consecutive detections to re-lock.
            # A threshold of 1 allowed a single noisy S&P spike to trigger premature re-lock.
            if self.consecutive_detections >= 2:
                self.state = TrackingState.LOCKED
                self.t_reacquired = timestamp
                if self.t_loss_start is not None:
                    self.last_reacquisition_time_sec = self.t_reacquired - self.t_loss_start

        return self.state

    def is_locked(self) -> bool:
        return self.state == TrackingState.LOCKED
