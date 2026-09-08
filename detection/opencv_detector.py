import cv2
import numpy as np
from typing import Optional
from frame.schema import Frame
from tracking.observation import CentroidObservation
from detection.base import BaseDetector

class OpenCVBeaconDetector(BaseDetector):
    """
    FR-14 Beacon Detection & FR-15 Centroid Estimation (Baseline Implementation).
    Uses binary thresholding + connected component analysis + intensity-weighted
    image moments to estimate the sub-pixel (x, y) centroid of the optical beacon.
    """
    def __init__(self, threshold_value: int = 200, min_area: float = 4.0, max_area: float = 400.0):
        self.threshold_value = threshold_value
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame: Frame) -> Optional[CentroidObservation]:
        img = frame.image

        # Convert to grayscale if image is BGR
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Step 1: Binary Thresholding
        _, binary = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)

        # Step 2: Connected Component Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        if num_labels <= 1:
            # Only background found
            return None

        # Filter candidate blobs by area (skip index 0, which is background)
        best_candidate_idx = None
        best_area = 0.0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_area <= area <= self.max_area:
                if area > best_area:
                    best_area = area
                    best_candidate_idx = i

        if best_candidate_idx is None:
            return None

        # Extract stats for the best blob
        x = stats[best_candidate_idx, cv2.CC_STAT_LEFT]
        y = stats[best_candidate_idx, cv2.CC_STAT_TOP]
        w = stats[best_candidate_idx, cv2.CC_STAT_WIDTH]
        h = stats[best_candidate_idx, cv2.CC_STAT_HEIGHT]
        bbox = (x, y, w, h)

        # Step 3: Intensity-Weighted Sub-Pixel Centroid Estimation (FR-15)
        roi_mask = (labels[y:y+h, x:x+w] == best_candidate_idx).astype(np.uint8)
        roi_gray = gray[y:y+h, x:x+w]
        moments = cv2.moments(roi_gray * roi_mask)

        if moments["m00"] > 0:
            cx = x + (moments["m10"] / moments["m00"])
            cy = y + (moments["m01"] / moments["m00"])
            confidence = min(1.0, float(moments["m00"] / (best_area * 255.0)) + 0.5)
        else:
            # Fallback to geometric centroid from stats if moments fail
            cx = centroids[best_candidate_idx][0]
            cy = centroids[best_candidate_idx][1]
            confidence = 0.8

        return CentroidObservation(
            centroid_x=float(cx),
            centroid_y=float(cy),
            confidence=float(confidence),
            bbox=bbox,
            area_px=float(best_area),
            is_valid=True
        )
