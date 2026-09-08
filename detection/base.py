from abc import ABC, abstractmethod
from typing import Optional
from frame.schema import Frame
from tracking.observation import CentroidObservation

class BaseDetector(ABC):
    """
    FR-14 & Section 7 Architectural Backbone: Swappable Beacon Detector Interface.
    Any detection algorithm (baseline thresholding or future research detector)
    must implement this exact interface.
    """
    @abstractmethod
    def detect(self, frame: Frame) -> Optional[CentroidObservation]:
        """
        Processes a Frame object and returns a CentroidObservation if beacon is detected,
        or None if no valid beacon target is found.
        """
        pass
