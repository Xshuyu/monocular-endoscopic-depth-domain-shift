"""Comparable PCA and robust-covariance OBB estimators for validation."""

from __future__ import annotations

import numpy as np
from .lie_algebra_obb import compute_robust_obb_lie


def _sorted_extent(points: np.ndarray, rotation: np.ndarray, quantiles=None) -> np.ndarray:
    centered = points - np.median(points, axis=0)
    local = centered @ rotation
    if quantiles is None:
        ext = np.ptp(local, axis=0)
    else:
        lo, hi = quantiles
        ext = np.quantile(local, hi, axis=0) - np.quantile(local, lo, axis=0)
    return np.sort(ext)[::-1]


def pca_obb(points: np.ndarray, quantiles=None) -> np.ndarray:
    """PCA axes; report full extent unless quantiles are explicitly supplied."""
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(points - center, rowvar=False))
    rotation = vecs[:, ::-1]
    return _sorted_extent(points, rotation, quantiles)


def mad_trimmed_pca_obb(points: np.ndarray, mad_scale: float = 3.5, quantiles=(0.01, 0.99)) -> np.ndarray:
    """Use MAD only for orientation, then use a stated extent quantile.

    This separates robust axis estimation from the endpoint definition and
    avoids silently trimming the lesion boundary from the reported diameter.
    """
    points = np.asarray(points, dtype=float)
    center = np.median(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    med = np.median(radii)
    mad = np.median(np.abs(radii - med))
    # Match the radial filter used by compute_robust_obb_lie exactly.
    keep = radii <= med + mad_scale * max(mad, 1e-4)
    if np.sum(keep) < 10:
        keep = np.ones(len(points), dtype=bool)
    _, vecs = np.linalg.eigh(np.cov(points[keep] - np.mean(points[keep], axis=0), rowvar=False))
    rotation = vecs[:, ::-1]
    return _sorted_extent(points, rotation, quantiles=quantiles)


def mad_filtered_pca_obb(points: np.ndarray, mad_scale: float = 3.5, quantiles=(0.01, 0.99)) -> np.ndarray:
    """PCA orientation after the same radial MAD filter used by the SO(3) OBB."""
    points = np.asarray(points, dtype=float)
    center = np.median(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    med = np.median(radii)
    mad = np.median(np.abs(radii - med))
    threshold = med + mad_scale * max(mad, 1e-4)
    keep = radii <= threshold
    if np.sum(keep) < 10:
        keep = np.ones(len(points), dtype=bool)
    filtered = points[keep]
    _, vecs = np.linalg.eigh(np.cov(filtered - np.mean(filtered, axis=0), rowvar=False))
    rotation = vecs[:, ::-1]
    return _sorted_extent(filtered, rotation, quantiles=quantiles)


def pca_orientation_disagreement_deg(points: np.ndarray, mad_scale: float = 3.5) -> float:
    """Angle between full PCA and MAD-filtered PCA major-axis directions."""
    points = np.asarray(points, dtype=float)
    center = np.median(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    med = np.median(radii)
    mad = np.median(np.abs(radii - med))
    threshold = med + mad_scale * max(mad, 1e-4)
    keep = radii <= threshold
    if np.sum(keep) < 10:
        return 0.0
    _, full_vec = np.linalg.eigh(np.cov(points - np.mean(points, axis=0), rowvar=False))
    _, robust_vec = np.linalg.eigh(np.cov(points[keep] - np.mean(points[keep], axis=0), rowvar=False))
    v0 = full_vec[:, -1]
    v1 = robust_vec[:, -1]
    cosine = np.clip(abs(float(np.dot(v0, v1))), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def adaptive_pca_obb(points: np.ndarray, disagreement_threshold_deg: float, mad_scale: float = 3.5):
    """Protect clean-case full extents; invoke MAD-filtered full-support PCA only
    when full and robust principal directions disagree beyond a validation-set
    threshold. Returns (extents, diagnostics).
    """
    points = np.asarray(points, dtype=float)
    angle = pca_orientation_disagreement_deg(points, mad_scale=mad_scale)
    if angle > disagreement_threshold_deg:
        center = np.median(points, axis=0)
        radii = np.linalg.norm(points - center, axis=1)
        med = np.median(radii)
        mad = np.median(np.abs(radii - med))
        keep = radii <= med + mad_scale * max(mad, 1e-4)
        if np.sum(keep) < 10:
            keep = np.ones(len(points), dtype=bool)
        subset = points[keep]
        center_sub = np.mean(subset, axis=0)
        _, vecs = np.linalg.eigh(np.cov(subset - center_sub, rowvar=False))
        local = (subset - center_sub) @ vecs[:, ::-1]
        ext = np.sort(np.ptp(local, axis=0))[::-1]
        return ext, {"mode": "mad_filtered", "axis_disagreement_deg": angle, "retained_fraction": float(np.mean(keep))}
    return pca_obb(points), {"mode": "pca_full_support", "axis_disagreement_deg": angle, "retained_fraction": 1.0}


def mad_rejection_fraction(points: np.ndarray, mad_scale: float = 3.5) -> float:
    points = np.asarray(points, dtype=float)
    center = np.median(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    med = np.median(radii)
    mad = np.median(np.abs(radii - med))
    keep = radii <= med + mad_scale * max(mad, 1e-4)
    if np.sum(keep) < 10:
        return 0.0
    return float(np.mean(~keep))


def adaptive_filter_obb(points: np.ndarray, rejection_threshold: float):
    """Use full-support PCA unless radial MAD rejection exceeds a locked threshold."""
    fraction = mad_rejection_fraction(points)
    if fraction > rejection_threshold:
        result = compute_robust_obb_lie(points, outlier_filter="always")
        ext = np.array([result.major_diameter_mm, result.minor_diameter_mm, result.elevation_height_mm])
        return ext, {"mode": "mad_filtered_so3", "rejection_fraction": fraction}
    return pca_obb(points), {"mode": "pca_full_support", "rejection_fraction": fraction}


def confidence_aware_obb(points: np.ndarray, low_threshold: float, high_threshold: float):
    """Return a size estimate only in interpretable contamination regimes.

    Low rejection fraction: use PCA full-support. High rejection fraction:
    use MAD-filtered OBB. Between thresholds, abstain from a single-point size
    and return a disagreement interval from the two estimators.
    """
    if not 0 <= low_threshold < high_threshold < 1:
        raise ValueError("Require 0 <= low_threshold < high_threshold < 1")
    rejection = mad_rejection_fraction(points)
    pca = pca_obb(points)
    robust = compute_robust_obb_lie(points, outlier_filter="always")
    robust_size = np.array([robust.major_diameter_mm, robust.minor_diameter_mm, robust.elevation_height_mm])
    if rejection < low_threshold:
        return {"status": "quantify_pca", "estimate_mm": pca, "interval_mm": None, "rejection_fraction": rejection}
    if rejection > high_threshold:
        return {"status": "quantify_robust", "estimate_mm": robust_size, "interval_mm": None, "rejection_fraction": rejection}
    low = np.minimum(pca, robust_size)
    high = np.maximum(pca, robust_size)
    return {"status": "abstain_interval", "estimate_mm": None, "interval_mm": [low, high], "rejection_fraction": rejection}


def huber_irls_pca_obb(points: np.ndarray, delta: float = 1.345, max_iter: int = 50, quantiles=(0.01, 0.99)) -> np.ndarray:
    """Robust covariance directions via Huber IRLS and stated extent quantiles."""
    points = np.asarray(points, dtype=float)
    center = np.median(points, axis=0)
    centered = points - center
    cov = np.cov(centered, rowvar=False) + np.eye(3) * 1e-8
    for _ in range(max_iter):
        inv = np.linalg.pinv(cov)
        md = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", centered, inv, centered), 1e-12))
        scale = max(float(np.median(md) / 1.538), 1e-8)
        u = md / scale
        weights = np.minimum(1.0, delta / np.maximum(u, 1e-12))
        new_center = np.average(points, axis=0, weights=weights)
        residual = points - new_center
        new_cov = (residual * weights[:, None]).T @ residual / np.sum(weights)
        new_cov += np.eye(3) * 1e-8
        if np.linalg.norm(new_cov - cov) / max(np.linalg.norm(cov), 1e-8) < 1e-7:
            cov = new_cov
            break
        center, cov = new_center, new_cov
    _, vecs = np.linalg.eigh(cov)
    return _sorted_extent(points, vecs[:, ::-1], quantiles=quantiles)
