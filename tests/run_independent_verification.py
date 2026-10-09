"""Independent, audit-traceable verification of MetricEndo components.

This suite intentionally does not treat Kvasir-SEG pseudo-millimetre outputs
or the legacy synthetic image cohort as clinical ground truth. It validates
the camera equations, OBB estimators on known point clouds, forceps scale
equation under pixel quantization, and curvature estimator on analytic data.
"""

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist

from lesion_3d_measurement.src.camera_model import CameraIntrinsics, backproject_depth_to_pointcloud, project_pointcloud_to_image
from lesion_3d_measurement.src.differential_geometry import compute_manifold_curvature
from lesion_3d_measurement.src.lie_algebra_obb import compute_robust_obb_lie
from lesion_3d_measurement.src.robust_obb_methods import pca_obb, mad_trimmed_pca_obb, mad_filtered_pca_obb, huber_irls_pca_obb
from lesion_3d_measurement.src.metric_scale import MetricScaleCalibrator, ToolDimensionPrior


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "independent_verification"
SEED = 4102026


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values, seed=SEED, draws=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))]


def stable_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pca_extents(points, quantiles=None):
    center = points.mean(axis=0)
    centered = points - center
    _, vecs = np.linalg.eigh(np.cov(centered, rowvar=False))
    coords = centered @ vecs[:, ::-1]
    if quantiles is None:
        ext = np.ptp(coords, axis=0)
    else:
        ext = np.quantile(coords, quantiles[1], axis=0) - np.quantile(coords, quantiles[0], axis=0)
    return np.sort(ext)[::-1]


def sample_ellipsoid_surface(rng, axes, n):
    """Sample radial directions and intersect each ray with an ellipsoid."""
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    axes = np.asarray(axes, dtype=float)
    ray_scale = np.sqrt(np.sum((directions / axes[None, :]) ** 2, axis=1))
    return directions / ray_scale[:, None]


def experiment_projection():
    rng = np.random.default_rng(SEED)
    intr = CameraIntrinsics.from_fov(640, 480, 140)
    depths = rng.uniform(15, 70, (480, 640)).astype(np.float32)
    mask = np.zeros_like(depths, dtype=np.uint8)
    cv2.ellipse(mask, (320, 240), (45, 30), 23, 0, 360, 255, -1)
    xyz, uv = backproject_depth_to_pointcloud(depths, intr, mask)
    uv2, valid = project_pointcloud_to_image(xyz, intr)
    err = np.max(np.abs(uv[valid] - uv2[valid]))
    assert err < 1e-3, f"projection roundtrip failed: {err}"
    return {"n_points": int(len(xyz)), "max_reprojection_error_px": float(err)}


def experiment_obb_invariants():
    rng = np.random.default_rng(SEED + 777)
    base = sample_ellipsoid_surface(rng, [12.0, 7.0, 3.0], 4000)
    rotation = Rotation.random(random_state=rng)
    points = rotation.apply(base) + np.array([0.0, 0.0, 30.0])
    result = compute_robust_obb_lie(points)
    c = result.obb_corners_3d
    edges = np.array([
        np.linalg.norm(c[1] - c[0]),
        np.linalg.norm(c[3] - c[0]),
        np.linalg.norm(c[4] - c[0]),
    ])
    reported = np.array([result.major_diameter_mm, result.minor_diameter_mm, result.elevation_height_mm])
    assert np.allclose(np.sort(edges)[::-1], reported, atol=1e-8)
    rotated = Rotation.random(random_state=rng).apply(points)
    rotated_result = compute_robust_obb_lie(rotated)
    rotation_error = np.max(np.abs(reported - np.array([
        rotated_result.major_diameter_mm, rotated_result.minor_diameter_mm, rotated_result.elevation_height_mm
    ])))
    assert np.isclose(np.linalg.det(result.rotation_matrix), 1.0, atol=1e-8)
    return {
        "reported_vs_corner_edge_max_abs_error_mm": float(np.max(np.abs(np.sort(edges)[::-1] - reported))),
        "rotation_invariance_max_abs_error_mm": float(rotation_error),
        "rotation_invariance_pass_0_01mm": bool(rotation_error <= 0.01),
        "clean_ellipsoid_major_error_mm": float(abs(result.major_diameter_mm - 24.0)),
        "clean_ellipsoid_minor_error_mm": float(abs(result.minor_diameter_mm - 14.0)),
        "clean_ellipsoid_thickness_error_mm": float(abs(result.elevation_height_mm - 6.0)),
        "det_rotation": float(np.linalg.det(result.rotation_matrix)),
    }


def experiment_obb(n_trials=200, n_surface=4000):
    rng = np.random.default_rng(SEED + 1)
    true_axes = np.array([12.0, 7.0, 3.0])
    truth_full = 2 * true_axes
    rows = []
    for trial in range(n_trials):
        local = sample_ellipsoid_surface(rng, true_axes, n_surface)
        if np.max(np.abs(local[:, 0])) > true_axes[0] * 1.001 or np.max(np.abs(local[:, 1])) > true_axes[1] * 1.001 or np.max(np.abs(local[:, 2])) > true_axes[2] * 1.001:
            raise AssertionError("Ellipsoid sampler exceeded analytical support bounds")
        rot = Rotation.random(random_state=rng)
        points = rot.apply(local) + rng.uniform([-4, -4, 25], [4, 4, 40])
        ref_rng = np.random.default_rng(SEED + 100_000 + trial)
        ref_local = sample_ellipsoid_surface(ref_rng, true_axes, 250_000)
        truth_q = np.sort(np.quantile(ref_local, .99, axis=0) - np.quantile(ref_local, .01, axis=0))[::-1]
        # For each noise level generate independent uniform spatial contamination.
        for noise_pct in (0, 2, 5, 10, 20):
            n_out = int(round(n_surface * noise_pct / 100.0))
            if n_out:
                out = rng.uniform([-40, -40, 5], [40, 40, 70], (n_out, 3))
                sample = np.vstack([points, out])
            else:
                sample = points.copy()
            estimates = {
                "pca_full_support": pca_obb(sample, quantiles=None),
                "pca_q01_q99": pca_obb(sample, quantiles=(0.01, 0.99)),
                "mad_orientation_q01_q99": mad_trimmed_pca_obb(sample, quantiles=(0.01, 0.99)),
                "mad_filtered_pca_q01_q99": mad_filtered_pca_obb(sample, quantiles=(0.01, 0.99)),
                "huber_irls_orientation_q01_q99": huber_irls_pca_obb(sample, quantiles=(0.01, 0.99)),
            }
            for filter_mode in ("never", "always"):
                robust = compute_robust_obb_lie(sample, outlier_filter=filter_mode)
                estimates[f"so3_qvolume_{filter_mode}_mad"] = np.array([
                    robust.major_diameter_mm, robust.minor_diameter_mm, robust.elevation_height_mm
                ])
            for method, ext in estimates.items():
                estimated = np.sort(ext)[::-1]
                endpoint = "full_support" if method == "pca_full_support" else "q01_q99_support"
                target = truth_full if endpoint == "full_support" else truth_q
                err = float(np.mean(np.abs(estimated - target)))
                rows.append({
                    "trial": trial, "noise_pct": noise_pct, "method": method,
                    "endpoint_estimand": endpoint,
                    "true_major_mm": truth_full[0], "true_minor_mm": truth_full[1], "true_thickness_mm": truth_full[2],
                    "reference_major_mm": float(target[0]), "reference_minor_mm": float(target[1]), "reference_thickness_mm": float(target[2]),
                    "estimated_major_mm": float(estimated[0]),
                    "estimated_minor_mm": float(estimated[1]),
                    "estimated_thickness_mm": float(estimated[2]),
                    "mean_abs_extent_error_mm": err,
                })
    write_csv(OUT / "obb_trials.csv", rows)
    summary = []
    for noise in (0, 2, 5, 10, 20):
        for method in ("pca_full_support", "pca_q01_q99", "mad_orientation_q01_q99", "mad_filtered_pca_q01_q99", "huber_irls_orientation_q01_q99", "so3_qvolume_never_mad", "so3_qvolume_always_mad"):
            e = np.array([r["mean_abs_extent_error_mm"] for r in rows if r["noise_pct"] == noise and r["method"] == method])
            summary.append({
                "noise_pct": noise, "method": method, "n_trials": len(e),
                "mean_mae_mm": float(e.mean()), "sd_mm": float(e.std(ddof=1)),
                "bootstrap_95ci": bootstrap_ci(e, SEED + noise + len(method)),
                "median_mae_mm": float(np.median(e)),
            })
    return summary


def experiment_tool_scale():
    rng = np.random.default_rng(SEED + 2)
    intr = CameraIntrinsics.from_fov(640, 480, 140)
    calibrator = MetricScaleCalibrator()
    rows = []
    for distance in (20.0, 30.0, 45.0):
        ideal_width = intr.fx * 2.4 / distance
        for width_error in (-0.2, -0.1, 0.0, 0.1, 0.2):
            for trial in range(100):
                observed = max(2, int(round(ideal_width * (1 + width_error))))
                # Build a full connected shaft mask with exactly observed pixels.
                mask = np.zeros((480, 640), np.uint8)
                x0 = int(rng.integers(100, 500 - observed))
                y0 = int(rng.integers(30, 140))
                mask[y0:450, x0:x0 + observed] = 255
                # Relative depth is constructed so the true tool distance is known.
                rel = np.full((480, 640), distance / 30.0, np.float32)
                scale, pred_z = calibrator.calibrate_with_tool(rel, mask, intr, ToolDimensionPrior(2.4))
                target_scale = 30.0
                rows.append({
                    "distance_mm": distance, "nominal_width_error_pct": width_error * 100,
                    "observed_width_px": observed, "trial": trial,
                    "estimated_tool_depth_mm": float(pred_z), "target_tool_depth_mm": distance,
                    "scale_factor": float(scale), "target_scale_factor": target_scale,
                    "depth_abs_error_mm": abs(float(pred_z) - distance),
                    "scale_abs_error": abs(float(scale) - target_scale),
                })
    write_csv(OUT / "tool_scale_trials.csv", rows)
    summary = []
    for distance in (20.0, 30.0, 45.0):
        subset = [r for r in rows if r["distance_mm"] == distance]
        err = np.array([r["depth_abs_error_mm"] for r in subset])
        summary.append({
            "distance_mm": distance, "n_trials": len(err), "mean_depth_mae_mm": float(err.mean()),
            "sd_mm": float(err.std(ddof=1)), "bootstrap_95ci": bootstrap_ci(err, SEED + int(distance)),
        })
    return summary


def sphere_points(radius, n_lat=31, n_lon=61):
    # Visible sphere cap (0--60 degrees from the viewing axis), matching the
    # local Monge-patch assumption rather than a full closed sphere.
    phi = np.linspace(0.02, np.pi / 3, n_lat)
    theta = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    p, t = np.meshgrid(phi, theta, indexing="ij")
    return np.column_stack([
        (radius * np.sin(p) * np.cos(t)).ravel(),
        (radius * np.sin(p) * np.sin(t)).ravel(),
        (radius * np.cos(p)).ravel(),
    ])


def experiment_curvature():
    rows = []
    for radius in (5.0, 10.0, 20.0):
        base = sphere_points(radius)
        normal = base / radius
        for noise in (0.0, 0.005, 0.01):
            for seed in range(20):
                rng = np.random.default_rng(SEED + 1000 + seed)
                points = base + normal * rng.normal(0, noise, (len(base), 1))
                profile = compute_manifold_curvature(
                    points,
                    radius * (1 - np.cos(np.pi / 3)),
                    k_neighbors=12,
                    normal_orientation="outward",
                    random_seed=SEED + seed,
                )
                rows.append({
                    "surface": "sphere", "radius_mm": radius, "normal_noise_sd_mm": noise,
                    "seed": seed, "expected_abs_H": 1 / radius,
                    "estimated_abs_H": abs(profile.mean_mean_curvature_H),
                    "expected_K": 1 / radius**2, "estimated_K": profile.mean_gaussian_curvature_K,
                    "abs_H_error": abs(abs(profile.mean_mean_curvature_H) - 1 / radius),
                    "abs_K_error": abs(profile.mean_gaussian_curvature_K - 1 / radius**2),
                })
    write_csv(OUT / "curvature_trials.csv", rows)
    summary = []
    for radius in (5.0, 10.0, 20.0):
        for noise in (0.0, 0.005, 0.01):
            subset = [r for r in rows if r["radius_mm"] == radius and r["normal_noise_sd_mm"] == noise]
            h_err = np.array([r["abs_H_error"] for r in subset])
            k_err = np.array([r["abs_K_error"] for r in subset])
            summary.append({
                "radius_mm": radius, "normal_noise_sd_mm": noise, "n_trials": len(subset),
                "mean_abs_H_error": float(h_err.mean()), "mean_abs_K_error": float(k_err.mean()),
                "H_error_ci": bootstrap_ci(h_err, SEED + int(radius * 10 + noise * 1e5)),
                "K_error_ci": bootstrap_ci(k_err, SEED + int(radius * 20 + noise * 1e5)),
            })
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "seed": SEED,
            "obb_trials_per_condition": 200,
            "tool_scale_trials_per_condition": 100,
            "curvature_trials_per_condition": 20,
            "claim_boundary": "component-level synthetic/analytic validation only; no clinical accuracy",
        },
        "projection_roundtrip": experiment_projection(),
        "obb_invariants": experiment_obb_invariants(),
        "obb_summary": experiment_obb(),
        "tool_scale_summary": experiment_tool_scale(),
        "curvature_summary": experiment_curvature(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
        },
    }
    path = OUT / "verification_summary.json"
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
