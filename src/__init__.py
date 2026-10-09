"""
3D Lesion Measurement Package.
"""

from .camera_model import CameraIntrinsics, backproject_depth_to_pointcloud, project_pointcloud_to_image
from .metric_scale import MetricScaleCalibrator, ToolDimensionPrior
from .differential_geometry import compute_manifold_curvature, ManifoldCurvatureProfile
from .lie_algebra_obb import compute_robust_obb_lie, RobustOBBResult
from .heat_geodesic import HeatGeodesicSolver
from .size_estimator import estimate_lesion_size_3d, LesionSizeResult, classify_risk_category
from .depth_estimator import BaseDepthEstimator, GeometricEndoscopyDepthEstimator
from .depth_anything_backbone import DepthAnythingV2Backbone
from .pipeline import Lesion3DMeasurementPipeline, MeasurementReport

__all__ = [
    "CameraIntrinsics",
    "backproject_depth_to_pointcloud",
    "project_pointcloud_to_image",
    "MetricScaleCalibrator",
    "ToolDimensionPrior",
    "compute_manifold_curvature",
    "ManifoldCurvatureProfile",
    "compute_robust_obb_lie",
    "RobustOBBResult",
    "HeatGeodesicSolver",
    "estimate_lesion_size_3d",
    "LesionSizeResult",
    "classify_risk_category",
    "BaseDepthEstimator",
    "GeometricEndoscopyDepthEstimator",
    "DepthAnythingV2Backbone",
    "Lesion3DMeasurementPipeline",
    "MeasurementReport"
]
