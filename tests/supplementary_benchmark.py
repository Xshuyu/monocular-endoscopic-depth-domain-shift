"""Supplementary controlled experiments for the CBM revision.

All experiments use analytical or procedurally generated ground truth.  The
script writes per-condition CSV files and a JSON summary so reported numbers
are traceable to individual trials.
"""

import csv
import json
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np

from lesion_3d_measurement.src.camera_model import CameraIntrinsics, backproject_depth_to_pointcloud
from lesion_3d_measurement.src.differential_geometry import compute_manifold_curvature
from lesion_3d_measurement.src.lie_algebra_obb import compute_robust_obb_lie
from lesion_3d_measurement.src.metric_scale import MetricScaleCalibrator, ToolDimensionPrior
from lesion_3d_measurement.src.size_estimator import estimate_lesion_size_3d


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "supplementary"


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ellipsoid(seed, major=24.0, minor=14.0, height=6.0, n=500):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, np.pi, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    rad = rng.uniform(0, 1, n) ** (1 / 3)
    return np.column_stack([
        major / 2 * rad * np.sin(phi) * np.cos(theta),
        minor / 2 * rad * np.sin(phi) * np.sin(theta),
        height / 2 * rad * np.cos(phi) + 30,
    ])


def experiment_robustness():
    rows = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        inliers = _ellipsoid(seed)
        for noise_pct in (0, 5, 10, 15, 20):
            n_out = int(len(inliers) * noise_pct / 100)
            outliers = rng.uniform([-40, -40, 10], [40, 40, 50], (n_out, 3))
            points = np.vstack([inliers, outliers])
            centered = points - np.mean(points, axis=0)
            _, vecs = np.linalg.eigh(np.cov(centered, rowvar=False))
            pca_ext = np.ptp(centered @ vecs, axis=0)
            pca_major = float(np.max(pca_ext))
            robust = compute_robust_obb_lie(points)
            rows.append({
                "seed": seed, "noise_pct": noise_pct,
                "pca_major_mm": pca_major,
                "robust_major_mm": robust.major_diameter_mm,
                "pca_abs_error_mm": abs(pca_major - 24.0),
                "robust_abs_error_mm": abs(robust.major_diameter_mm - 24.0),
            })
    _write_csv(OUT / "robustness_trials.csv", rows)
    summary = []
    for noise in (0, 5, 10, 15, 20):
        subset = [r for r in rows if r["noise_pct"] == noise]
        summary.append({
            "noise_pct": noise,
            "pca_mae_mm": float(np.mean([r["pca_abs_error_mm"] for r in subset])),
            "robust_mae_mm": float(np.mean([r["robust_abs_error_mm"] for r in subset])),
            "pca_sd_mm": float(np.std([r["pca_abs_error_mm"] for r in subset], ddof=1)),
            "robust_sd_mm": float(np.std([r["robust_abs_error_mm"] for r in subset], ddof=1)),
        })
    return summary


def experiment_scale_and_tool():
    """Measure scale bias under distance, tool width error, and mask perturbation."""
    intrinsics = CameraIntrinsics.from_fov(640, 480, 140.0)
    calibrator = MetricScaleCalibrator()
    rows = []
    for distance in (20.0, 30.0, 45.0):
        for width_bias in (-0.2, -0.1, 0.0, 0.1, 0.2):
            for mask_mode in ("clean", "eroded", "dilated"):
                h, w = 480, 640
                yy, xx = np.mgrid[:h, :w]
                true_tool_width = intrinsics.fx * 2.4 / distance
                observed_width = max(2, int(round(true_tool_width * (1 + width_bias))))
                tool = np.zeros((h, w), dtype=np.uint8)
                tool[120:430, 320:320 + observed_width] = 255
                if mask_mode == "eroded":
                    tool = cv2.erode(tool, np.ones((3, 3), np.uint8), iterations=2)
                elif mask_mode == "dilated":
                    tool = cv2.dilate(tool, np.ones((3, 3), np.uint8), iterations=2)
                gt_depth = np.full((h, w), distance, dtype=np.float32)
                relative = gt_depth / 30.0
                scale, tool_depth = calibrator.calibrate_with_tool(
                    relative, tool, intrinsics, ToolDimensionPrior(2.4)
                )
                target_scale = distance / float(np.median(relative[tool > 0]))
                rows.append({
                    "distance_mm": distance,
                    "width_bias_pct": width_bias * 100,
                    "mask_mode": mask_mode,
                    "estimated_tool_depth_mm": tool_depth,
                    "scale_factor": scale,
                    "target_scale": target_scale,
                    "relative_scale_error_pct": abs(scale - target_scale) / target_scale * 100,
                })
    _write_csv(OUT / "scale_tool_trials.csv", rows)
    return {
        "mean_relative_scale_error_pct": float(np.mean([r["relative_scale_error_pct"] for r in rows])),
        "max_relative_scale_error_pct": float(np.max([r["relative_scale_error_pct"] for r in rows])),
        "by_condition": rows,
    }


def _sphere(radius, n=25):
    u = np.linspace(0, np.pi / 3, n)
    v = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([
        (radius * np.outer(np.sin(u), np.cos(v))).ravel(),
        (radius * np.outer(np.sin(u), np.sin(v))).ravel(),
        (radius * np.outer(np.cos(u), np.ones_like(v))).ravel() + 30,
    ])


def experiment_curvature():
    rows = []
    for radius in (5.0, 10.0, 20.0):
        for seed in range(5):
            base = _sphere(radius)
            normal = base - np.array([0.0, 0.0, 30.0])
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            points = base + normal * np.random.default_rng(seed).normal(0, 0.01, (len(base), 1))
            profile = compute_manifold_curvature(points, radius * (1 - np.cos(np.pi / 3)), k_neighbors=12)
            rows.append({
                "surface": "sphere", "radius_mm": radius, "seed": seed,
                "expected_abs_H": 1 / radius, "estimated_abs_H": abs(profile.mean_mean_curvature_H),
                "expected_K": 1 / radius ** 2, "estimated_K": profile.mean_gaussian_curvature_K,
                "abs_H_error": abs(abs(profile.mean_mean_curvature_H) - 1 / radius),
                "abs_K_error": abs(profile.mean_gaussian_curvature_K - 1 / radius ** 2),
            })
    for noise in (0.0, 0.01, 0.05, 0.1):
        rng = np.random.default_rng(int(noise * 1000) + 7)
        x = np.linspace(-6, 6, 25)
        X, Y = np.meshgrid(x, x)
        base = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        noisy = base.copy()
        noisy[:, 2] += rng.normal(0, noise, len(noisy))
        profile = compute_manifold_curvature(noisy, 0.0, k_neighbors=12)
        rows.append({
            "surface": "plane", "radius_mm": "NA", "seed": int(noise * 1000),
            "expected_abs_H": 0.0, "estimated_abs_H": abs(profile.mean_mean_curvature_H),
            "expected_K": 0.0, "estimated_K": profile.mean_gaussian_curvature_K,
            "abs_H_error": abs(profile.mean_mean_curvature_H),
            "abs_K_error": abs(profile.mean_gaussian_curvature_K),
        })
    _write_csv(OUT / "curvature_trials.csv", rows)
    return rows


def experiment_pipeline_scale_effect():
    """Compare full and prior scale on known synthetic depth without RGB inference."""
    rows = []
    for distance in (20.0, 30.0, 45.0):
        for seed in range(10):
            rng = np.random.default_rng(seed)
            h, w = 240, 320
            intrinsics = CameraIntrinsics.from_fov(w, h, 140.0)
            yy, xx = np.mgrid[:h, :w]
            mask = (((xx - w / 2) / 35) ** 2 + ((yy - h / 2) / 25) ** 2) < 1
            depth = np.full((h, w), distance, dtype=np.float32)
            depth[mask] -= 3.0
            relative = depth / 30.0
            tool = np.zeros((h, w), dtype=np.uint8)
            tool[100:230, 250:255] = 255
            calibrator = MetricScaleCalibrator()
            tool_scale, _ = calibrator.calibrate_with_tool(relative, tool, intrinsics, ToolDimensionPrior(2.4))
            prior_scale = calibrator.calibrate_with_prior(relative, target_distance_mm=30.0)
            expected_major = 70.0 * distance / intrinsics.fx
            for name, scale in (("tool", tool_scale), ("prior", prior_scale)):
                points, _ = backproject_depth_to_pointcloud(relative * scale, intrinsics, mask.astype(np.uint8))
                result = estimate_lesion_size_3d(points, use_lie_algebra=True, use_heat_method=False)
                rows.append({
                    "distance_mm": distance, "seed": seed, "scale_mode": name,
                    "estimated_major_mm": result.major_diameter_mm,
                    "major_error_mm": abs(result.major_diameter_mm - expected_major),
                })
    _write_csv(OUT / "pipeline_scale_trials.csv", rows)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "robustness_summary": experiment_robustness(),
        "scale_tool": experiment_scale_and_tool(),
        "curvature_trials": experiment_curvature(),
        "pipeline_scale_trials": experiment_pipeline_scale_effect(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
    }
    with (OUT / "supplementary_summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({
        "output": str(OUT / "supplementary_summary.json"),
        "robustness": payload["robustness_summary"],
        "scale_error_mean_pct": payload["scale_tool"]["mean_relative_scale_error_pct"],
    }, indent=2))


if __name__ == "__main__":
    main()
