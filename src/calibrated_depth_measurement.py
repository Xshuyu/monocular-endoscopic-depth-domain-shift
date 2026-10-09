"""Experimental calibration and measurement; no claim of SOTA or clinical validity."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import isotonic_regression

@dataclass
class MonotoneDepthCalibration:
    x: np.ndarray
    y: np.ndarray

    @classmethod
    def fit(cls, prediction, depth_mm, *, increasing, bins=64):
        x, y = np.asarray(prediction, float).ravel(), np.asarray(depth_mm, float).ravel()
        if x.shape != y.shape or len(x) < bins:
            raise ValueError('Matched calibration arrays with enough samples required')
        if not (np.isfinite(x).all() and np.isfinite(y).all() and (y > 0).all()):
            raise ValueError('Calibration inputs must be finite and depth positive')
        order = np.argsort(x, kind='stable')
        groups = np.array_split(order, bins)
        centers = np.array([np.median(x[g]) for g in groups])
        values = np.array([np.median(y[g]) for g in groups])
        weights = np.array([len(g) for g in groups], float)
        # Merge tied centers before weighted PAVA to keep interpolation well-defined.
        unique, inv = np.unique(centers, return_inverse=True)
        total = np.bincount(inv, weights=weights)
        values = np.bincount(inv, weights=values*weights)/total
        values = isotonic_regression(values, weights=total, increasing=increasing).x
        return cls(unique, values)

    def predict(self, prediction):
        a = np.asarray(prediction)
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite prediction')
        return np.clip(np.interp(a, self.x, self.y), 1., 100.)

    def to_dict(self):
        return {'x': self.x.tolist(), 'y': self.y.tolist()}


def measure_axial_depth(depth_mm, mask, calibrated_rays):
    """MAD-filtered PCA 1--99% extent, NOT full caliper/geodesic diameter.

    Rays must come from the actual calibrated camera (including distortion),
    have shape H,W,3 and positive Z. Axial depth Z is not ray distance.
    No default FOV is supplied. Caller is responsible for ray calibration.
    """
    z = np.asarray(depth_mm, float)
    mask = np.asarray(mask, bool)
    rays = np.asarray(calibrated_rays, float)
    if z.ndim != 2 or mask.shape != z.shape or rays.shape != (*z.shape, 3):
        raise ValueError('Mismatched depth, mask or ray dimensions')
    if not mask.any():
        raise ValueError('Empty lesion mask')
    valid = mask & np.isfinite(z) & (z > 0) & np.isfinite(rays).all(axis=-1) & (rays[..., 2] > 0)
    if valid.sum() < 10:
        raise ValueError('Insufficient valid lesion support')
    points = rays[valid] / rays[valid, 2, None] * z[valid, None]
    center = np.median(points, axis=0)
    distance = np.linalg.norm(points-center, axis=1)
    med = np.median(distance)
    mad = np.median(np.abs(distance-med))
    keep = distance <= med + 3.*1.4826*max(mad, np.finfo(float).eps)
    clean = points[keep]
    if len(clean) < 10:
        raise ValueError('Insufficient support after filtering')
    _, _, axes = np.linalg.svd(clean-clean.mean(axis=0), full_matrices=False)
    coordinates = (clean-clean.mean(axis=0)) @ axes.T
    extents = np.quantile(coordinates, .99, axis=0)-np.quantile(coordinates, .01, axis=0)
    return {'major_q01_q99_mm': float(extents.max()),
            'extents_mm': sorted(extents.tolist(), reverse=True),
            'valid_mask_fraction': float(valid.sum()/mask.sum()),
            'retained_point_fraction': float(keep.mean()),
            'endpoint': 'MAD-PCA 1--99% extent; not clinical caliper diameter'}
