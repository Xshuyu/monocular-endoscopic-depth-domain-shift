"""
Monocular Endoscopy Depth Estimation Interface.
Supports depth models (e.g. Depth Anything v2 / MiDaS) and geometric endoscopic prior.
"""

from typing import Optional
import numpy as np
import cv2


class BaseDepthEstimator:
    """Abstract interface for monocular depth estimation."""
    def predict(self, rgb_image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GeometricEndoscopyDepthEstimator(BaseDepthEstimator):
    """
    Geometric endoscopy prior depth estimator.
    Models the lumen tunnel geometry (depth increases towards the dark intestinal lumen)
    and incorporates photometric shading cues for local mucosal elevation.
    Useful for offline testing and baseline geometric comparison.
    """

    def __init__(self, min_depth_mm: float = 12.0, max_depth_mm: float = 65.0):
        self.min_depth_mm = min_depth_mm
        self.max_depth_mm = max_depth_mm

    def predict(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Estimates relative depth based on photometric and distance-from-center cues.
        """
        h, w = rgb_image.shape[:2]
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) if len(rgb_image.shape) == 3 else rgb_image
        gray_norm = cv2.GaussianBlur(gray, (21, 21), 0).astype(np.float32) / 255.0

        # Endoscopic lumen cue: lumen center is usually darker and deeper
        # Inverted illumination: darker areas correspond to deeper lumen
        depth_photometric = (1.0 - gray_norm)

        # Distance from optical axis: edges of cylindrical lumen are closer
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        dist_norm = dist_from_center / np.max(dist_from_center)

        # Combine tunnel geometry (central lumen deeper) with mucosal shading
        depth_raw = 0.6 * depth_photometric + 0.4 * (1.0 - dist_norm)
        depth_raw = np.clip(depth_raw, 0.05, 1.0)

        # Normalize to target depth range (mm)
        d_min, d_max = np.min(depth_raw), np.max(depth_raw)
        depth_mm = self.min_depth_mm + (depth_raw - d_min) / (d_max - d_min + 1e-6) * (self.max_depth_mm - self.min_depth_mm)

        return depth_mm.astype(np.float32)
