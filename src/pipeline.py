"""
End-to-End 3D Lesion Physical Measurement Pipeline (Mathematically Enhanced).
Integrates depth estimation, metric scale calibration, 3D back-projection,
Lie group SO(3) robust manifold OBB, Laplace-Beltrami Heat Geodesics,
Weingarten differential curvature shape index, and AR medical visualization overlay.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from .camera_model import CameraIntrinsics, backproject_depth_to_pointcloud, project_pointcloud_to_image
from .metric_scale import MetricScaleCalibrator, ToolDimensionPrior
from .size_estimator import estimate_lesion_size_3d, LesionSizeResult
from .depth_estimator import BaseDepthEstimator, GeometricEndoscopyDepthEstimator


@dataclass
class MeasurementReport:
    result: LesionSizeResult
    metric_depth_map: np.ndarray
    annotated_image: np.ndarray
    scale_factor: float
    used_tool_calibration: bool


class Lesion3DMeasurementPipeline:
    """
    Complete clinical pipeline for measuring 3D polyp/lesion physical dimensions from monocular endoscopy.
    """

    def __init__(
        self,
        intrinsics: Optional[CameraIntrinsics] = None,
        depth_estimator: Optional[BaseDepthEstimator] = None,
        fov_deg: float = 140.0,
        use_lie_algebra: bool = True,
        use_heat_method: bool = True
    ):
        self.fov_deg = fov_deg
        self.intrinsics = intrinsics
        self.depth_estimator = depth_estimator or GeometricEndoscopyDepthEstimator()
        self.scale_calibrator = MetricScaleCalibrator()
        self.use_lie_algebra = use_lie_algebra
        self.use_heat_method = use_heat_method

    def process(
        self,
        rgb_image: np.ndarray,
        lesion_mask: np.ndarray,
        tool_mask: Optional[np.ndarray] = None,
        tool_prior: ToolDimensionPrior = ToolDimensionPrior()
    ) -> MeasurementReport:
        """
        Executes end-to-end 3D measurement for a single endoscopy frame.

        Args:
            rgb_image: RGB image, shape (H, W, 3), dtype uint8.
            lesion_mask: Binary lesion mask (H, W), non-zero indicates lesion pixels.
            tool_mask: Optional binary mask of instrument in the field of view.
            tool_prior: Physical tool dimension specification.

        Returns:
            MeasurementReport containing dimensions, physical depth map, and visualization.
        """
        h, w = rgb_image.shape[:2]

        # 1. Initialize camera intrinsics if not specified
        if self.intrinsics is None or self.intrinsics.width != w or self.intrinsics.height != h:
            intrinsics = CameraIntrinsics.from_fov(width=w, height=h, fov_deg=self.fov_deg)
        else:
            intrinsics = self.intrinsics

        # 2. Estimate relative depth
        raw_depth = self.depth_estimator.predict(rgb_image)

        # 3. Metric Scale Calibration
        used_tool = False
        if tool_mask is not None and np.sum(tool_mask > 0) > 50:
            scale_factor, _, _ = self.scale_calibrator.calibrate_adaptive(
                relative_depth_map=raw_depth,
                tool_mask=tool_mask,
                intrinsics=intrinsics,
                tool_prior=tool_prior,
            )
            used_tool = True
        else:
            scale_factor = self.scale_calibrator.calibrate_with_prior(
                relative_depth_map=raw_depth,
                roi_mask=lesion_mask
            )

        metric_depth = raw_depth * scale_factor

        # 4. Back-project lesion mask to 3D physical point cloud
        lesion_pts_3d, coords_2d = backproject_depth_to_pointcloud(
            depth_map=metric_depth,
            intrinsics=intrinsics,
            mask=lesion_mask
        )

        if len(lesion_pts_3d) < 4:
            raise ValueError("Lesion mask contains too few valid depth points for 3D measurement.")

        # 5. Measure 3D Physical Dimensions with Mathematical Enhancements
        size_result = estimate_lesion_size_3d(
            points_3d=lesion_pts_3d,
            use_lie_algebra=self.use_lie_algebra,
            use_heat_method=self.use_heat_method
        )

        # 6. Generate Augmented Reality (AR) Overlay
        annotated_img = self._render_ar_overlay(
            rgb_image=rgb_image,
            lesion_mask=lesion_mask,
            size_result=size_result,
            intrinsics=intrinsics
        )

        return MeasurementReport(
            result=size_result,
            metric_depth_map=metric_depth,
            annotated_image=annotated_img,
            scale_factor=scale_factor,
            used_tool_calibration=used_tool
        )

    def render_3d_dashboard(
        self,
        rgb_image: np.ndarray,
        lesion_mask: np.ndarray,
        report: MeasurementReport,
        output_path: str,
        mode: str = "clinical"
    ) -> np.ndarray:
        """
        Renders an ultra-high-definition 3D multi-view medical dashboard.
        """
        from .visualization_3d import render_3d_measurement_dashboard
        h, w = rgb_image.shape[:2]
        intrinsics = self.intrinsics if self.intrinsics else CameraIntrinsics.from_fov(w, h, self.fov_deg)
        return render_3d_measurement_dashboard(
            rgb_image=rgb_image,
            lesion_mask=lesion_mask,
            metric_depth_map=report.metric_depth_map,
            size_result=report.result,
            intrinsics=intrinsics,
            output_path=output_path,
            mode=mode
        )

    def _render_ar_overlay(
        self,
        rgb_image: np.ndarray,
        lesion_mask: np.ndarray,
        size_result: LesionSizeResult,
        intrinsics: CameraIntrinsics
    ) -> np.ndarray:
        """
        Renders an ergonomic, zero-occlusion medical visualization:
        - Left Panel: Pristine endoscopic view with ONLY clean, ultra-thin boundary contour (0% texture occlusion).
        - Right Panel: Dedicated clinical telemetry dashboard displaying 3D OBB wireframe,
                       millimeter metrics, differential geometry profile, and guideline stratification.
        """
        h, w = rgb_image.shape[:2]
        panel_w = 340  # Dedicated right-hand clinical telemetry sidebar

        # 1. Prepare pristine/clean endoscopic image (Left Canvas)
        # Avoid heavy solid/dark masks: use only a delicate 2px glowing boundary so mucosal textures are 100% visible
        left_view = rgb_image.copy()
        mask_u8 = (lesion_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Subtle translucent rim highlight (barely 12% opacity)
        rim_overlay = np.zeros_like(left_view)
        cv2.drawContours(rim_overlay, contours, -1, (0, 255, 120), thickness=cv2.FILLED)
        left_view = cv2.addWeighted(left_view, 0.88, rim_overlay, 0.12, 0)
        # Crisp lesion boundary
        cv2.drawContours(left_view, contours, -1, (0, 255, 180), 2, cv2.LINE_AA)

        # Draw projected 3D OBB wireframe with fine golden lines
        corners_2d, valid = project_pointcloud_to_image(size_result.obb_corners, intrinsics)
        if np.all(valid):
            pts = corners_2d.astype(int)
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
            for start, end in edges:
                pt1 = tuple(pts[start])
                pt2 = tuple(pts[end])
                cv2.line(left_view, pt1, pt2, (255, 215, 0), 1, cv2.LINE_AA)

        # 2. Construct Dedicated Clinical Telemetry Sidebar (Right Canvas)
        right_panel = np.full((h, panel_w, 3), 24, dtype=np.uint8)  # Modern dark slate background
        # Divider line
        cv2.line(right_panel, (0, 0), (0, h), (60, 65, 75), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        curv = size_result.curvature_profile

        # Header Title
        cv2.putText(right_panel, "MetricEndo 3D Telemetry", (18, 32), font, 0.58, (0, 255, 200), 2, cv2.LINE_AA)
        cv2.line(right_panel, (18, 42), (panel_w - 18, 42), (80, 90, 100), 1)

        # Section 1: Physical Metrology
        cv2.putText(right_panel, "[ PHYSICAL METROLOGY ]", (18, 68), font, 0.44, (160, 175, 190), 1, cv2.LINE_AA)
        metrics_1 = [
            ("Major Axis (L):", f"{size_result.major_diameter_mm:.2f} mm", (0, 255, 255)),
            ("Minor Axis (W):", f"{size_result.minor_diameter_mm:.2f} mm", (240, 240, 240)),
            ("Elevation (H):", f"{size_result.elevation_height_mm:.2f} mm", (240, 240, 240)),
            ("Heat Geodesic:", f"{size_result.geodesic_length_mm:.2f} mm", (0, 220, 255)),
            ("Surface Area:", f"{size_result.surface_area_mm2:.1f} mm2", (200, 210, 220)),
            ("Est. Volume:", f"{size_result.volume_mm3:.1f} mm3", (200, 210, 220)),
        ]
        y_pos = 92
        for label, val, color in metrics_1:
            cv2.putText(right_panel, label, (22, y_pos), font, 0.42, (180, 185, 190), 1, cv2.LINE_AA)
            cv2.putText(right_panel, val, (160, y_pos), font, 0.44, color, 1, cv2.LINE_AA)
            y_pos += 22

        cv2.line(right_panel, (18, y_pos), (panel_w - 18, y_pos), (50, 55, 65), 1)
        y_pos += 24

        # Section 2: Differential Manifold Curvature
        cv2.putText(right_panel, "[ MANIFOLD CURVATURE ]", (18, y_pos), font, 0.44, (160, 175, 190), 1, cv2.LINE_AA)
        y_pos += 24
        metrics_2 = [
            ("Mean Curv (H):", f"{curv.mean_mean_curvature_H:+.3f} mm-1"),
            ("Gauss Curv (K):", f"{curv.mean_gaussian_curvature_K:+.4f} mm-2"),
            ("Curvedness (C):", f"{curv.mean_curvedness:.3f}"),
            ("Shape Index (S):", f"{curv.mean_shape_index:+.2f}"),
        ]
        for label, val in metrics_2:
            cv2.putText(right_panel, label, (22, y_pos), font, 0.42, (180, 185, 190), 1, cv2.LINE_AA)
            cv2.putText(right_panel, val, (160, y_pos), font, 0.44, (220, 225, 230), 1, cv2.LINE_AA)
            y_pos += 22

        cv2.line(right_panel, (18, y_pos), (panel_w - 18, y_pos), (50, 55, 65), 1)
        y_pos += 24

        # Section 3: Clinical Stratification Card
        cv2.putText(right_panel, "[ CLINICAL CLASSIFICATION ]", (18, y_pos), font, 0.44, (160, 175, 190), 1, cv2.LINE_AA)
        y_pos += 26
        # Paris Morphology Tag
        cv2.putText(right_panel, "Paris Morph:", (22, y_pos), font, 0.42, (180, 185, 190), 1, cv2.LINE_AA)
        cv2.putText(right_panel, size_result.paris_morphology[:20], (22, y_pos + 18), font, 0.42, (0, 255, 150), 1, cv2.LINE_AA)
        y_pos += 46

        # Guideline Risk Tag
        cv2.putText(right_panel, "Guideline Risk:", (22, y_pos), font, 0.42, (180, 185, 190), 1, cv2.LINE_AA)
        risk_short = size_result.risk_category.split("(")[0].strip()
        risk_color = (0, 140, 255) if "Giant" in risk_short or "Advanced" in risk_short else (0, 255, 180)
        cv2.putText(right_panel, risk_short, (22, y_pos + 18), font, 0.46, risk_color, 2, cv2.LINE_AA)

        # 3. Concatenate horizontally: 100% original endoscopic field + right clinical telemetry panel
        composite_view = np.hstack([left_view, right_panel])
        return composite_view
