"""MAD-trimmed quantile-volume OBB optimization over the SO(3) manifold.

The estimator uses a robust radial prefilter, 2nd--98th percentile extents
for orientation fitting, and 1st--99th percentile extents for reported size.
It is not a Huber M-estimator; ``huber_loss`` is retained only as a standalone
utility and is not used by the OBB optimizer.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class RobustOBBResult:
    """
    Robust 3D Oriented Bounding Box obtained via Lie algebra optimization.
    """
    major_diameter_mm: float      # Optimized major axis length
    minor_diameter_mm: float      # Optimized minor axis length
    elevation_height_mm: float    # Optimized normal thickness/elevation
    rotation_matrix: np.ndarray   # 3x3 orthonormal rotation matrix R in SO(3)
    center: np.ndarray            # 3D centroid [X, Y, Z] in mm
    obb_corners_3d: np.ndarray    # 8x3 physical corners in mm
    residual_error: float         # Huber loss value at convergence


def hat(v: np.ndarray) -> np.ndarray:
    """
    Lie algebra hat operator: maps 3-vector v to 3x3 skew-symmetric matrix v^.
    """
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """
    Exponential map from Lie algebra so(3) vector to Lie group SO(3) rotation matrix
    using Rodrigues' rotation formula.
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)

    axis = omega / theta
    k = hat(axis)
    # R = I + sin(theta)*K + (1 - cos(theta))*K^2
    R = np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * np.dot(k, k)
    return R


def huber_loss(x: np.ndarray, delta: float = 1.5) -> Tuple[float, np.ndarray]:
    """
    Huber loss function and its derivative (weights for M-estimation).
    """
    abs_x = np.abs(x)
    is_linear = abs_x > delta
    loss = np.where(is_linear, delta * (abs_x - 0.5 * delta), 0.5 * (x ** 2))
    # Weight w = psi(x) / x for iteratively reweighted least squares
    weights = np.where(abs_x < 1e-8, 1.0, np.where(is_linear, delta / abs_x, 1.0))
    return float(np.sum(loss)), weights


def compute_robust_obb_lie(
    points_3d: np.ndarray,
    max_iters: int = 40,
    lr: float = 0.05,
    delta: float = 1.2,
    boundary_correction: float = 0.0,
    adaptive_boundary: bool = False,
    outlier_filter: str = "adaptive",
) -> RobustOBBResult:
    """Estimate a trimmed-quantile OBB by minimizing coordinate-box volume.

    Args:
        points_3d: (N, 3) point cloud in camera coordinate frame.
        max_iters: Maximum gradient iterations on the Lie algebra tangent space.
        lr: Learning rate on tangent space.
        delta: Huber threshold for robust outlier clipping.

    Returns:
        RobustOBBResult instance.
    """
    pts = points_3d.astype(np.float64)
    n_pts = len(pts)

    # 0. Detect gross radial contamination. A radial MAD rule can reject valid
    # ellipsoid support points even for clean, uniformly sampled surfaces; only
    # activate it when the tail is inconsistent with the robust radial core.
    center = np.median(pts, axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    mad = np.median(np.abs(dists - np.median(dists)))
    threshold = np.median(dists) + 3.5 * max(mad, 1e-4)
    inlier_mask = dists <= threshold
    radial_tail = float(np.mean(~inlier_mask))
    if outlier_filter not in {"adaptive", "always", "never"}:
        raise ValueError("outlier_filter must be adaptive, always, or never")
    apply_filter = outlier_filter == "always" or (outlier_filter == "adaptive" and radial_tail > 0.01)
    if outlier_filter == "never":
        apply_filter = False
    pts_clean = pts[inlier_mask] if apply_filter and np.sum(inlier_mask) >= 10 else pts

    center = np.median(pts_clean, axis=0)
    centered = pts_clean - center
    cov = np.cov(centered, rowvar=False)
    _, eig_vecs = np.linalg.eigh(cov)

    # Ensure right-handed orthonormal coordinate frame (det(R) = +1)
    R = eig_vecs[:, [2, 1, 0]]  # Order by descending eigenvalue
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]

    def objective(rotation):
        local = centered @ rotation
        ext = np.percentile(local, 98.0, axis=0) - np.percentile(local, 2.0, axis=0)
        # Log-volume is dimensionless in its gradient scale and avoids steps
        # growing cubically with lesion size.
        return float(np.sum(np.log(np.maximum(ext, 1e-6))))

    # Deterministic starts: PCA frame plus fixed small rotations. This reduces
    # the chance that a non-convex volume objective remains at a poor local
    # orientation while keeping runtime bounded and repeatable.
    starts = [R.copy()]
    for axis in np.eye(3):
        for angle in (-np.pi / 4, np.pi / 4, -np.pi / 2, np.pi / 2):
            starts.append(R @ so3_exp(axis * angle))
    step_size = min(max(float(lr), 1e-3), 0.1)
    eps = 1e-4
    best_R = starts[0]
    best_objective = objective(best_R)

    # Riemannian finite-difference descent with monotone backtracking from each
    # deterministic start. The best converged objective is retained.
    for start in starts:
        current = start.copy()
        local_step = step_size
        for _ in range(max_iters):
            base_obj = objective(current)
            grad = np.zeros(3, dtype=np.float64)
            for axis in range(3):
                delta = np.zeros(3, dtype=np.float64)
                delta[axis] = eps
                grad[axis] = (objective(current @ so3_exp(delta)) - base_obj) / eps
            grad_norm = float(np.linalg.norm(grad))
            if not np.isfinite(grad_norm) or grad_norm < 1e-7:
                break
            direction = -grad / grad_norm
            accepted = False
            alpha = local_step
            for _line_search in range(12):
                candidate = current @ so3_exp(direction * alpha)
                if objective(candidate) < base_obj - 1e-10:
                    current = candidate
                    local_step = min(alpha * 1.15, 0.1)
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                break
        local_objective = objective(current)
        if local_objective < best_objective:
            best_objective = local_objective
            best_R = current.copy()

    # Keep a valid right-handed orthonormal basis despite floating point drift.
    u, _, vt = np.linalg.svd(best_R)
    best_R = u @ vt
    if np.linalg.det(best_R) < 0:
        u[:, -1] *= -1
        best_R = u @ vt

    # 3. Final Extents and Corners Calculation with best R
    R_opt = best_R
    pts_canon = np.dot(centered, R_opt)

    min_bounds = np.percentile(pts_canon, 1.0, axis=0)
    max_bounds = np.percentile(pts_canon, 99.0, axis=0)
    final_extents = max_bounds - min_bounds
    if adaptive_boundary:
        # Activate correction only when the robust core indicates substantial
        # tail loss.  This avoids applying the ellipsoid correction to flat
        # plaques and saddles where it creates a positive bias.
        core_width = np.percentile(pts_canon, 95.0, axis=0) - np.percentile(pts_canon, 5.0, axis=0)
        support_width = np.ptp(pts_canon, axis=0)
        tail_ratio = np.max((support_width - core_width) / np.maximum(support_width, 1e-6))
        boundary_correction = boundary_correction * float(np.clip((tail_ratio - 0.12) / 0.18, 0.0, 1.0))
    if boundary_correction > 0.0:
        # Estimate the omitted tails from the robust core.  The correction
        # is a validation-set parameter, never fitted on the test sample.
        q05 = np.percentile(pts_canon, 5.0, axis=0)
        q95 = np.percentile(pts_canon, 95.0, axis=0)
        core_extents = q95 - q05
        tail_width = np.maximum(core_extents - final_extents, 0.0)
        # The 5--95% core is smaller than the 1--99% measurement in normal
        # data.  Use the empirical full support as the correction reservoir.
        support_width = np.ptp(pts_canon, axis=0)
        tail_width = np.maximum(support_width - final_extents, tail_width)
        final_extents = final_extents + boundary_correction * tail_width

    # Sort extents so [0] is major, [1] is minor, [2] is elevation height
    ext_sorted_indices = np.argsort(final_extents)[::-1]
    final_extents_sorted = final_extents[ext_sorted_indices]
    R_opt = R_opt[:, ext_sorted_indices]

    # Maintain det(R) = 1
    if np.linalg.det(R_opt) < 0:
        R_opt[:, 2] = -R_opt[:, 2]

    major_d = float(final_extents_sorted[0])
    minor_d = float(final_extents_sorted[1])
    elevation_h = float(final_extents_sorted[2])

    # 4. Construct 8 OBB corners from the exact reported trimmed bounds.
    # Apply the same axis permutation used for extents and correction.
    bounds_center = 0.5 * (min_bounds + max_bounds)
    reordered_center = bounds_center[ext_sorted_indices]
    corrected_half = 0.5 * final_extents_sorted
    min_b = reordered_center - corrected_half
    max_b = reordered_center + corrected_half

    corners_canonical = np.array([
        [min_b[0], min_b[1], min_b[2]],
        [max_b[0], min_b[1], min_b[2]],
        [max_b[0], max_b[1], min_b[2]],
        [min_b[0], max_b[1], min_b[2]],
        [min_b[0], min_b[1], max_b[2]],
        [max_b[0], min_b[1], max_b[2]],
        [max_b[0], max_b[1], max_b[2]],
        [min_b[0], max_b[1], max_b[2]],
    ])

    obb_corners_3d = np.dot(corners_canonical, R_opt.T) + center

    # Derive the reported extents from the final output corners so that the
    # dimensions and visualized OBB cannot silently disagree.
    edge_lengths = np.array([
        np.linalg.norm(obb_corners_3d[1] - obb_corners_3d[0]),
        np.linalg.norm(obb_corners_3d[3] - obb_corners_3d[0]),
        np.linalg.norm(obb_corners_3d[4] - obb_corners_3d[0]),
    ])
    final_extents_sorted = edge_lengths[np.argsort(edge_lengths)[::-1]]
    major_d, minor_d, elevation_h = map(float, final_extents_sorted)

    return RobustOBBResult(
        major_diameter_mm=major_d,
        minor_diameter_mm=minor_d,
        elevation_height_mm=elevation_h,
        rotation_matrix=R_opt,
        center=center,
        obb_corners_3d=obb_corners_3d,
        residual_error=float(np.exp(best_objective))
    )
