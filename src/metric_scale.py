"""
Metric Scale Recovery Module for Monocular Endoscopy.
Recovers absolute physical depth (in millimeters) from relative depth predictions.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
from .camera_model import CameraIntrinsics


@dataclass
class ToolDimensionPrior:
    """
    Physical dimension priors for common endoscopic instruments (in millimeters).
    """
    name: str = "Standard Biopsy Forceps"
    sheath_diameter_mm: float = 2.4     # Typical outer diameter: 2.4 mm (for 2.8mm working channel)
    open_jaw_span_mm: float = 6.0       # Typical open cup width: 6.0 mm - 7.2 mm


class MetricScaleCalibrator:
    """
    Calculates the metric scaling factor to map relative depth to physical millimeters.
    """

    def __init__(self, default_working_distance_mm: float = 25.0):
        self.default_working_distance_mm = default_working_distance_mm

    def calibrate_with_tool(
        self,
        relative_depth_map: np.ndarray,
        tool_mask: np.ndarray,
        intrinsics: CameraIntrinsics,
        tool_prior: ToolDimensionPrior = ToolDimensionPrior()
    ) -> Tuple[float, float]:
        """
        Calibrate metric scale using instrument shaft width as a visual reference.

        Args:
            relative_depth_map: 2D array of relative depth (arbitrary positive unit).
            tool_mask: Binary mask of the endoscopic tool (H, W).
            intrinsics: CameraIntrinsics instance.
            tool_prior: Physical dimensions of the tool.

        Returns:
            scale_factor (s): Multiplier where Depth_metric = s * Depth_relative.
            tool_depth_mm: Estimated physical distance to the tool.
        """
        tool_mask_uint8 = (tool_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(tool_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback to working distance prior
            scale = self.calibrate_with_prior(relative_depth_map)
            return scale, self.default_working_distance_mm

        # Select the largest contour (main instrument body)
        largest_c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_c)
        (_, (w_box, h_box), _) = rect

        # The smaller dimension of the bounding box roughly corresponds to the shaft width
        tool_pixel_width = min(w_box, h_box)
        if tool_pixel_width < 1.0:
            scale = self.calibrate_with_prior(relative_depth_map)
            return scale, self.default_working_distance_mm

        # Average focal length
        f_avg = (intrinsics.fx + intrinsics.fy) / 2.0

        # Physical pinhole geometry:
        # tool_pixel_width = (f_avg * tool_diameter_mm) / Z_tool_mm
        # Z_tool_mm = (f_avg * tool_diameter_mm) / tool_pixel_width
        tool_depth_mm = (f_avg * tool_prior.sheath_diameter_mm) / tool_pixel_width

        # Get median relative depth at the tool region
        tool_rel_depths = relative_depth_map[tool_mask > 0]
        tool_rel_depths = tool_rel_depths[tool_rel_depths > 0]
        if len(tool_rel_depths) == 0:
            scale = self.calibrate_with_prior(relative_depth_map)
            return scale, self.default_working_distance_mm

        rel_depth_median = float(np.median(tool_rel_depths))
        if rel_depth_median <= 1e-4:
            scale = 1.0
        else:
            # Physical scaling factor s = Z_true / Z_pred
            scale = tool_depth_mm / rel_depth_median

        return float(scale), float(tool_depth_mm)

    def calibrate_with_prior(
        self,
        relative_depth_map: np.ndarray,
        roi_mask: Optional[np.ndarray] = None,
        target_distance_mm: Optional[float] = None
    ) -> float:
        """
        Calibrate scale factor based on statistical working distance prior.
        """
        target_dist = target_distance_mm if target_distance_mm is not None else self.default_working_distance_mm
        if roi_mask is not None and np.sum(roi_mask > 0) > 10:
            valid_vals = relative_depth_map[roi_mask > 0]
        else:
            valid_vals = relative_depth_map[relative_depth_map > 0]

        if len(valid_vals) == 0:
            return 1.0

        median_rel = float(np.median(valid_vals))
        if median_rel <= 1e-4:
            return 1.0

        return float(target_dist / median_rel)

    def estimate_tool_quality(
        self,
        tool_mask: np.ndarray,
        relative_depth_map: np.ndarray,
    ) -> float:
        """Estimate whether a tool mask is reliable enough to affect scale.

        The score is deliberately deterministic and uses only image-space
        evidence. Thin, fragmented, or depth-inconsistent masks receive a
        low score, causing the caller to rely more on the working-distance
        prior.
        """
        mask = (tool_mask > 0).astype(np.uint8)
        area = int(np.sum(mask))
        if area < 50:
            return 0.0
        contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        short_edge = min(width, height)
        long_edge = max(width, height)
        if short_edge < 2.0 or long_edge / max(short_edge, 1e-6) < 3.0:
            return 0.0
        component_ratio = cv2.contourArea(contour) / max(float(area), 1.0)
        depths = relative_depth_map[mask > 0]
        depth_cv = float(np.std(depths) / max(np.mean(depths), 1e-6)) if len(depths) else 1.0
        width_score = min(short_edge / 8.0, 1.0)
        compactness_score = min(component_ratio, 1.0)
        depth_score = float(np.exp(-4.0 * depth_cv))
        return float(np.clip(width_score * compactness_score * depth_score, 0.0, 1.0))

    def calibrate_adaptive(
        self,
        relative_depth_map: np.ndarray,
        tool_mask: Optional[np.ndarray],
        intrinsics: CameraIntrinsics,
        tool_prior: ToolDimensionPrior = ToolDimensionPrior(),
        target_distance_mm: float = 30.0,
    ) -> Tuple[float, float, float]:
        """Fuse tool and prior scales using a data-quality-dependent weight."""
        prior_scale = self.calibrate_with_prior(
            relative_depth_map, target_distance_mm=target_distance_mm
        )
        if tool_mask is None:
            return float(prior_scale), 0.0, float(prior_scale)
        tool_scale, _ = self.calibrate_with_tool(
            relative_depth_map, tool_mask, intrinsics, tool_prior
        )
        quality = self.estimate_tool_quality(tool_mask, relative_depth_map)
        final_scale = quality * tool_scale + (1.0 - quality) * prior_scale
        return float(final_scale), float(quality), float(prior_scale)
