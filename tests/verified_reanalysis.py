"""Reanalysis benchmark with separated scale and geometry factors.

Outputs are verification artifacts, not clinical validation. The script keeps
procedural truth, estimated-depth outputs, and unlabelled real-image outputs
separate and writes per-case predictions with deterministic uncertainty.
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
from scipy import stats

from lesion_3d_measurement.src.camera_model import CameraIntrinsics, backproject_depth_to_pointcloud
from lesion_3d_measurement.src.depth_anything_backbone import DepthAnythingV2Backbone
from lesion_3d_measurement.src.ground_truth_benchmark import generate_benchmark_cohort
from lesion_3d_measurement.src.lie_algebra_obb import compute_robust_obb_lie
from lesion_3d_measurement.src.metric_scale import MetricScaleCalibrator, ToolDimensionPrior
from lesion_3d_measurement.src.robust_obb_methods import pca_obb, mad_filtered_pca_obb


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "verified_reanalysis"
SEED = 20260924


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest():
    paths = [
        ROOT / "tests" / "verified_reanalysis.py",
        ROOT / "src" / "lie_algebra_obb.py",
        ROOT / "src" / "robust_obb_methods.py",
        ROOT / "src" / "metric_scale.py",
        ROOT / "src" / "evaluation_suite.py",
        ROOT / "src" / "ground_truth_benchmark.py",
        ROOT / "src" / "differential_geometry.py",
        ROOT / "src" / "heat_geodesic.py",
    ]
    products = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {
        "seed": SEED,
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
        "outputs": {str(p.relative_to(ROOT)): sha256(p) for p in products},
        "protocol": "verified_reanalysis.py; procedural factorial 50x4x4; bootstrap 10,000 draws; pairwise Wilcoxon with Holm adjustment",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(x, seed, draws=10000):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    sample = rng.choice(x, (draws, len(x)), replace=True).mean(axis=1)
    return [float(np.quantile(sample, .025)), float(np.quantile(sample, .975))]


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def summarize_errors(rows, group_key, group_values):
    methods = sorted({r["method"] for r in rows})
    table = []
    for value in group_values:
        for method in methods:
            subset = [r for r in rows if r[group_key] == value and r["method"] == method]
            if not subset:
                continue
            err = np.array([r["abs_error_mm"] for r in subset])
            signed = np.array([r["signed_error_mm"] for r in subset])
            table.append({
                group_key: value,
                "method": method,
                "n": len(err),
                "mae_mm": float(err.mean()),
                "mae_bootstrap_95ci": bootstrap_ci(err, SEED + int(float(value) * 17) + len(method)),
                "rmse_mm": float(np.sqrt(np.mean(signed ** 2))),
                "mre_pct": float(np.mean([r["relative_error_pct"] for r in subset])),
                "mean_bias_mm": float(signed.mean()),
                "risk_agreement_pct": float(100 * np.mean([r["risk_agreement"] for r in subset])),
            })
    return table


def risk_bin(value):
    return int(value >= 5) + int(value >= 10) + int(value >= 20)


def run_procedural_factorial():
    cohort = generate_benchmark_cohort(num_samples=50)
    intr = CameraIntrinsics.from_fov(640, 480, 140.0)
    depth_model = DepthAnythingV2Backbone(device="cuda")
    calibrator = MetricScaleCalibrator()
    methods = ("pca_full", "pca_q01_q99", "mad_pca_q01_q99", "so3_quantile_volume")
    scale_modes = ("oracle", "prior", "tool", "fusion")
    rows = []
    for sample in cohort:
        pred_rel = depth_model.predict(sample.rgb_image, tool_mask=sample.tool_mask)
        # A shape-preserving relative-depth oracle is used to isolate the
        # geometry stage from RGB depth-prediction and scale-recovery error.
        oracle_rel = sample.ground_truth_depth_mm / 30.0
        prior_scale = calibrator.calibrate_with_prior(pred_rel, roi_mask=sample.lesion_mask, target_distance_mm=30.0)
        tool_scale, _ = calibrator.calibrate_with_tool(pred_rel, sample.tool_mask, intr, ToolDimensionPrior(2.4))
        fusion_scale = 0.5 * prior_scale + 0.5 * tool_scale
        scale_maps = {
            "oracle": oracle_rel * 30.0,
            "prior": pred_rel * prior_scale,
            "tool": pred_rel * tool_scale,
            "fusion": pred_rel * fusion_scale,
        }
        true_risk = risk_bin(sample.true_major_mm)
        for scale_mode in scale_modes:
            points, _ = backproject_depth_to_pointcloud(scale_maps[scale_mode], intr, sample.lesion_mask)
            estimator_values = {
                "pca_full": pca_obb(points),
                "pca_q01_q99": pca_obb(points, quantiles=(.01, .99)),
                "mad_pca_q01_q99": mad_filtered_pca_obb(points, quantiles=(.01, .99)),
            }
            so3 = compute_robust_obb_lie(points)
            estimator_values["so3_quantile_volume"] = np.array([
                so3.major_diameter_mm, so3.minor_diameter_mm, so3.elevation_height_mm
            ])
            for method in methods:
                pred = float(np.sort(estimator_values[method])[::-1][0])
                err = pred - sample.true_major_mm
                rows.append({
                    "sample_id": sample.sample_id,
                    "morphology": sample.true_paris_morphology.split(" ")[1],
                    "working_distance_mm": sample.working_distance_mm,
                    "true_major_mm": sample.true_major_mm,
                    "scale_mode": scale_mode,
            "scale_factor": float({"oracle": 30.0, "prior": prior_scale, "tool": tool_scale, "fusion": fusion_scale}[scale_mode]),
                    "method": method,
                    "predicted_major_mm": pred,
                    "signed_error_mm": float(err),
                    "abs_error_mm": float(abs(err)),
                    "relative_error_pct": float(abs(err) / sample.true_major_mm * 100),
                    "risk_agreement": int(risk_bin(pred) == true_risk),
                })
    write_csv(OUT / "procedural_factorial_trials.csv", rows)
    summaries = []
    for scale in scale_modes:
        for method in methods:
            sub = [r for r in rows if r["scale_mode"] == scale and r["method"] == method]
            e = np.array([r["abs_error_mm"] for r in sub])
            summaries.append({
                "scale_mode": scale, "method": method, "n": len(e),
                "mae_mm": float(e.mean()), "mae_bootstrap_95ci": bootstrap_ci(e, SEED + len(scale) + len(method)),
                "rmse_mm": float(np.sqrt(np.mean([r["signed_error_mm"] ** 2 for r in sub]))),
                "mre_pct": float(np.mean([r["relative_error_pct"] for r in sub])),
                "risk_agreement_pct": float(100 * np.mean([r["risk_agreement"] for r in sub])),
            })
    paired = []
    for scale in scale_modes:
        by_method = {m: {r["sample_id"]: r["abs_error_mm"] for r in rows if r["scale_mode"] == scale and r["method"] == m} for m in methods}
        for method in methods[1:]:
            a = np.array([by_method[method][sid] for sid in sorted(by_method[method])])
            b = np.array([by_method["pca_full"][sid] for sid in sorted(by_method["pca_full"])])
            diff = a - b
            try:
                p = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            except ValueError:
                p = 1.0
            paired.append({
                "scale_mode": scale,
                "comparison": f"{method} vs pca_full",
                "mean_paired_abs_error_difference_mm": float(diff.mean()),
                "paired_difference_bootstrap_95ci": bootstrap_ci(diff, SEED + len(scale) + len(method)),
                "wilcoxon_p": p,
            })
    if paired:
        holm_p = holm_adjust([r["wilcoxon_p"] for r in paired])
        for row, adj in zip(paired, holm_p):
            row["wilcoxon_p_holm"] = float(adj)

    strata = []
    for morphology in sorted({r["morphology"] for r in rows}):
        for scale in scale_modes:
            for method in methods:
                sub = [r for r in rows if r["morphology"] == morphology and r["scale_mode"] == scale and r["method"] == method]
                error = np.array([r["abs_error_mm"] for r in sub])
                strata.append({
                    "morphology": morphology, "scale_mode": scale, "method": method,
                    "n": len(sub), "mae_mm": float(error.mean()),
                    "mae_bootstrap_95ci": bootstrap_ci(error, SEED + len(morphology) + len(scale) + len(method)),
                })
    scale_errors = {}
    for mode in scale_modes:
        err_by_id = {}
        for r in rows:
            if r["scale_mode"] == mode and r["method"] == "pca_full":
                err_by_id[r["sample_id"]] = r["signed_error_mm"]
        scale_errors[mode] = err_by_id
    ids = sorted(scale_errors["tool"])
    tool_err = np.array([scale_errors["tool"][i] for i in ids])
    prior_err = np.array([scale_errors["prior"][i] for i in ids])
    fused_err = np.array([scale_errors["fusion"][i] for i in ids])
    scale_error_diagnostics = {
        "tool_mean_signed_error_mm": float(tool_err.mean()),
        "prior_mean_signed_error_mm": float(prior_err.mean()),
        "fusion_mean_signed_error_mm": float(fused_err.mean()),
        "tool_prior_error_correlation": float(np.corrcoef(tool_err, prior_err)[0, 1]),
        "fusion_error_abs_less_than_both_fraction": float(np.mean((np.abs(fused_err) < np.abs(tool_err)) & (np.abs(fused_err) < np.abs(prior_err)))),
        "per_case_rows": [{"sample_id": i, "tool_signed_error_mm": float(scale_errors["tool"][i]), "prior_signed_error_mm": float(scale_errors["prior"][i]), "fusion_signed_error_mm": float(scale_errors["fusion"][i])} for i in ids],
    }
    return {"n_samples": len(cohort), "rows": len(rows), "summary": summaries, "paired_tests": paired, "morphology_strata": strata, "scale_error_diagnostics": scale_error_diagnostics}


def run_kvasir_pair_audit():
    base = ROOT / "datasets" / "kvasir-seg" / "Kvasir-SEG"
    images = {p.stem: p for p in (base / "images").glob("*.jpg")}
    masks = {p.stem: p for p in (base / "masks").glob("*.jpg")}
    common = sorted(images.keys() & masks.keys())
    rows = []
    for sample_id in common:
        image = cv2.imread(str(images[sample_id]))
        mask = cv2.imread(str(masks[sample_id]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2] != mask.shape[:2]:
            rows.append({"sample_id": sample_id, "status": "invalid_or_mismatched", "image_shape": None if image is None else image.shape[:2], "mask_shape": None if mask is None else mask.shape[:2]})
        else:
            rows.append({"sample_id": sample_id, "status": "matched", "image_shape": image.shape[:2], "mask_shape": mask.shape[:2], "mask_pixels": int(np.sum(mask > 0))})
    return {
        "n_images": len(images), "n_masks": len(masks), "n_matched_ids": len(common),
        "n_invalid_or_mismatched": sum(r["status"] != "matched" for r in rows),
        "sha256_images": sha256(Path(next(iter(images.values())))) if images else None,
        "paired_rows": rows,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "seed": SEED,
            "procedural_samples": 50,
            "factorial_design": "4 scale modes x 4 geometry estimators, same lesion mask/point support per sample",
            "bootstrap": "10,000 percentile draws over 50 paired procedural samples",
            "clinical_claim_boundary": "procedural algorithm comparison only; Kvasir-SEG used for pair audit/descriptive runtime, no physical size truth",
        },
        "procedural": run_procedural_factorial(),
        "kvasir_pair_audit": run_kvasir_pair_audit(),
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__,
        },
    }
    (OUT / "verified_reanalysis.json").write_text(json.dumps(payload, indent=2))
    write_manifest()
    print(json.dumps({
        "procedural_samples": payload["procedural"]["n_samples"],
        "factorial_rows": payload["procedural"]["rows"],
        "kvasir_pair_audit": {k: v for k, v in payload["kvasir_pair_audit"].items() if k != "paired_rows"},
        "summary": payload["procedural"]["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
