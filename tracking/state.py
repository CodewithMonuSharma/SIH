from enum import Enum

class TrackingState(Enum):
    """
    FR-20 Target Lock State & Section 5 State Machine.
    Defines system tracking states governing controller mode, search triggers,
    and performance metrics evaluation.
    """
    INITIALIZATION = "INITIALIZATION"
    SEARCHING = "SEARCHING"
    BEACON_DETECTED = "BEACON_DETECTED"
    ACQUISITION = "ACQUISITION"
    LOCKED = "LOCKED"
    TARGET_LOST = "TARGET_LOST"
    REACQUISITION = "REACQUISITION"
