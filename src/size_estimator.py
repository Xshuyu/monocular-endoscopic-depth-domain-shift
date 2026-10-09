"""
3D Lesion Physical Size and Geodesic Measurement Module (Mathematically Enhanced).
Integrates:
    - Lie group SO(3) / Lie algebra so(3) robust manifold optimization for OBB.
    - Heat Method on Laplace-Beltrami manifolds for continuous geodesic diameter.
    - Differential geometry (Weingarten shape operator, Gaussian & Mean curvature) for Paris morphology.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import cv2
from scipy.spatial import ConvexHull

from .lie_algebra_obb import compute_robust_obb_lie, RobustOBBResult
from .heat_geodesic import HeatGeodesicSolver
from .differential_geometry import compute_manifold_curvature, ManifoldCurvatureProfile
from .camera_model import backproject_depth_to_pointcloud


def estimate_with_mask_uncertainty(
    depth_map: np.ndarray,
    intrinsics,
    lesion_mask: np.ndarray,
    use_lie_algebra: bool = True,
    use_heat_method: bool = False,
    boundary_correction: float = 0.0,
    adaptive_boundary: bool = False,
) -> Dict[str, Any]:
    """Estimate size over eroded, original, and dilated lesion masks.

    The interval is a sensitivity interval, not a statistical confidence
    interval. It exposes segmentation-induced measurement variability.
    """
    mask = (lesion_mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    masks = {
        "eroded": np.where(cv2.erode(mask, kernel, iterations=1) > 0, 255, 0).astype(np.uint8),
        "original": np.where(mask > 0, 255, 0).astype(np.uint8),
        "dilated": np.where(cv2.dilate(mask, kernel, iterations=1) > 0, 255, 0).astype(np.uint8),
    }
    outputs = {}
    for name, candidate in masks.items():
        points, _ = backproject_depth_to_pointcloud(depth_map, intrinsics, candidate)
        if len(points) < 4:
            continue
        outputs[name] = estimate_lesion_size_3d(
            points,
            use_lie_algebra=use_lie_algebra,
            use_heat_method=use_heat_method,
            boundary_correction=boundary_correction,
        )
    if not outputs:
        raise ValueError("No valid mask variant contains enough depth points.")
    majors = np.array([v.major_diameter_mm for v in outputs.values()], dtype=float)
    return {
        "estimates": outputs,
        "major_median_mm": float(np.median(majors)),
        "major_interval_mm": [float(np.min(majors)), float(np.max(majors))],
        "major_sensitivity_mm": float(np.max(majors) - np.min(majors)),
    }


@dataclass
class LesionSizeResult:
    """
    Measurement results of a 3D lesion.
    All dimensions are in physical millimeters (mm) or cubic mm (mm³).
    """
    major_diameter_mm: float            # Primary clinical metric (长径, mm)
    minor_diameter_mm: float            # Secondary clinical metric (短径, mm)
    elevation_height_mm: float          # Lesion height from base (隆起高度, mm)
    geodesic_length_mm: float           # Heat Method continuous geodesic distance (曲面测地线长径, mm)
    surface_area_mm2: float             # Approximate physical surface area (表面积, mm²)
    volume_mm3: float                   # Approximate physical volume (体积, mm³)
    risk_category: str                  # ASGE/ESGE risk stratification category
    paris_morphology: str               # Paris clinical classification based on differential geometry
    curvature_profile: ManifoldCurvatureProfile  # Detailed Riemannian invariants
    center_3d: np.ndarray               # 3D centroid [X, Y, Z] in mm
    obb_corners: np.ndarray             # 8x3 corners of 3D Oriented Bounding Box


def classify_risk_category(major_diameter_mm: float) -> str:
    """
    Stratifies polyp according to clinical guidelines (ASGE/ESGE).
    """
    if major_diameter_mm < 5.0:
        return "Diminutive (< 5 mm, Low Risk, Non-Advanced)"
    elif 5.0 <= major_diameter_mm < 10.0:
        return "Small (5-9 mm, Intermediate Risk)"
    elif 10.0 <= major_diameter_mm < 20.0:
        return "Advanced Adenoma (10-19 mm, High Risk, Prompt Resection)"
    else:
        return "Giant / Complex Lesion (>= 20 mm, Very High Risk, Consider ESD/Surgery)"


def estimate_lesion_size_3d(
    points_3d: np.ndarray,
    use_lie_algebra: bool = True,
    use_heat_method: bool = True,
    boundary_correction: float = 0.0,
    adaptive_boundary: bool = False,
) -> LesionSizeResult:
    """
    Comprehensive 3D size estimation for a segmented lesion point cloud.

    Args:
        points_3d: (N, 3) numpy array of [X, Y, Z] in millimeters.
        use_lie_algebra: Whether to use SO(3) Lie manifold optimization for robust OBB.
        use_heat_method: Whether to use Laplace-Beltrami Heat Method for geodesic computation.

    Returns:
        LesionSizeResult instance with physical dimensions, Paris classification, and curvatures.
    """
    if len(points_3d) < 4:
        raise ValueError(f"Need at least 4 3D points, got {len(points_3d)}")

    # Keep the observed lesion boundary. Robust OBB performs its own explicit
    # MAD radial filtering; a blanket 95th-percentile radial trim here would
    # remove valid boundary points and unfairly alter the PCA ablation.
    filtered_pts = np.asarray(points_3d, dtype=np.float64)

    # 2. 3D Oriented Bounding Box via Lie Group SO(3) Manifold Optimization
    if use_lie_algebra:
        obb_res: RobustOBBResult = compute_robust_obb_lie(
            filtered_pts,
            boundary_correction=boundary_correction,
            adaptive_boundary=adaptive_boundary,
        )
        major_diameter = obb_res.major_diameter_mm
        minor_diameter = obb_res.minor_diameter_mm
        elevation_height = obb_res.elevation_height_mm
        center = obb_res.center
        obb_corners_3d = obb_res.obb_corners_3d
    else:
        # Fallback to PCA
        centered = filtered_pts - np.mean(filtered_pts, axis=0)
        cov = np.cov(centered, rowvar=False)
        _, eig_vecs = np.linalg.eigh(cov)
        projected = np.dot(centered, eig_vecs)
        extents = np.max(projected, axis=0) - np.min(projected, axis=0)
        ext_sorted = np.sort(extents)[::-1]
        major_diameter = float(ext_sorted[0])
        minor_diameter = float(ext_sorted[1])
        elevation_height = float(ext_sorted[2])
        center = np.mean(filtered_pts, axis=0)
        obb_corners_3d = np.zeros((8, 3))

    # 3. Geodesic distance along mucosal manifold via Heat Method
    if use_heat_method:
        heat_solver = HeatGeodesicSolver()
        geodesic_length = heat_solver.compute_manifold_geodesic_diameter(filtered_pts)
    else:
        # Euclidean upper bound fallback
        geodesic_length = major_diameter

    geodesic_length = max(geodesic_length, major_diameter)

    # 4. Differential Geometry & Paris Morphology via Weingarten Shape Operator
    curvature_profile = compute_manifold_curvature(
        points_3d=filtered_pts,
        elevation_height_mm=elevation_height
    )

    # 5. Approximate Area & Volume via Convex Hull or Ellipsoidal Model
    try:
        hull = ConvexHull(filtered_pts)
        surface_area = float(hull.area)
        volume = float(hull.volume)
    except Exception:
        a = major_diameter / 2.0
        b = minor_diameter / 2.0
        c = max(elevation_height, 0.5)
        surface_area = float(np.pi * a * b)
        volume = float((2.0 / 3.0) * np.pi * a * b * c)

    category = classify_risk_category(major_diameter)

    return LesionSizeResult(
        major_diameter_mm=major_diameter,
        minor_diameter_mm=minor_diameter,
        elevation_height_mm=elevation_height,
        geodesic_length_mm=geodesic_length,
        surface_area_mm2=surface_area,
        volume_mm3=volume,
        risk_category=category,
        paris_morphology=curvature_profile.morphology_class,
        curvature_profile=curvature_profile,
        center_3d=center,
        obb_corners=obb_corners_3d
    )
