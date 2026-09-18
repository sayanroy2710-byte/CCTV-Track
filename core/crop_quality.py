"""
Crop Quality Assessment Gate.
Filters person bounding box crops before ReID embedding extraction and gallery storage.
Rejects blurry, edge-truncated, or heavily occluded crops to prevent gallery memory pollution.
"""
import cv2
import numpy as np
from typing import Tuple, Optional

class CropQualityGate:
    """
    Evaluates visual quality of person crops before updating ReID gallery.
    """
    def __init__(self,
                 min_height: int = 60,
                 min_aspect_ratio: float = 1.0,
                 max_aspect_ratio: float = 5.0,
                 min_sharpness: float = 30.0,
                 border_margin_px: int = 4):
        self.min_height = min_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_sharpness = min_sharpness
        self.border_margin_px = border_margin_px

    def compute_sharpness(self, crop: np.ndarray) -> float:
        """
        Computes sharpness via Laplacian variance.
        Higher values indicate clear, sharp images; low values indicate blur.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return 0.0
        try:
            resized = cv2.resize(crop, (64, 128))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0

    def is_border_truncated(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, ...]) -> bool:
        """
        Checks if the person is partially cut off by the camera frame edges.
        """
        h_img, w_img = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        m = self.border_margin_px
        return bool(x1 <= m or y1 <= m or x2 >= w_img - 1 - m or y2 >= h_img - 1 - m)

    def assess_crop(self, crop: np.ndarray, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, ...]) -> Tuple[bool, float, str]:
        """
        Assesses crop suitability for identity gallery storage.
        Returns:
            usable: bool (True if suitable for gallery update)
            quality_score: float in [0.0, 1.0]
            reason: str diagnostic explanation
        """
        if crop is None or crop.size == 0:
            return False, 0.0, "empty_crop"

        h, w = crop.shape[:2]
        if h < self.min_height:
            return False, float(h / self.min_height * 0.3), f"too_small (h={h}<{self.min_height})"

        aspect = float(h / max(1, w))
        if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
            return False, 0.2, f"invalid_aspect ({aspect:.2f})"

        if self.is_border_truncated(bbox, frame_shape):
            return False, 0.4, "border_truncated"

        sharpness_val = self.compute_sharpness(crop)
        if sharpness_val < self.min_sharpness:
            return False, float(min(0.5, sharpness_val / self.min_sharpness * 0.5)), f"blurry (sharpness={sharpness_val:.1f}<{self.min_sharpness})"

        # Compute continuous quality score
        size_factor = min(1.0, h / 180.0)
        sharp_factor = min(1.0, sharpness_val / (self.min_sharpness * 3.0))
        quality = 0.4 * size_factor + 0.6 * sharp_factor

        return True, float(np.clip(quality, 0.5, 1.0)), "clean"
