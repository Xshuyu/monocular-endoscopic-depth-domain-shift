"""Scale-calibration verification on a geometrically consistent phantom.

The tool shaft and lesion center share known depth in the ideal condition.
Separate perturbations introduce axial offset, shaft tilt, pixel-width
quantization, and mask boundary errors. This is a formula verification, not a
clinical dataset evaluation.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import pearsonr

from lesion_3d_measurement.src.camera_model import CameraIntrinsics
from lesion_3d_measurement.src.metric_scale import MetricScaleCalibrator, ToolDimensionPrior


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "scale_phantom"
SEED = 20260925


def main():
    rng = np.random.default_rng(SEED)
    intr = CameraIntrinsics.from_fov(640, 480, 140.0)
    calibrator = MetricScaleCalibrator()
    rows = []
    for distance in (20.0, 30.0, 45.0):
        for axial_offset in (0.0, 1.0, 3.0, 5.0):
            for tilt_deg in (0.0, 15.0, 30.0, 45.0):
                for mask_edge_error_px in (-2, 0, 2):
                    for repeat in range(50):
                        tool_depth = distance + axial_offset
                        # Apparent width is reduced under shaft tilt relative to
                        # a cross-section perpendicular to the viewing axis.
                        ideal_px = intr.fx * 2.4 / tool_depth * np.cos(np.deg2rad(tilt_deg))
                        width = max(2, int(round(ideal_px)) + mask_edge_error_px)
                        x0 = int(rng.integers(150, 430))
                        y0 = int(rng.integers(40, 120))
                        mask = np.zeros((480, 640), np.uint8)
                        # Raster pixels represent pixel centres; cv2.minAreaRect
                        # spans centre-to-centre distance (n-1). Render one
                        # extra pixel so its measured geometric span equals the
                        # quantized projected shaft width used above.
                        mask[y0:440, x0:x0 + width + 1] = 255
                        relative_depth = np.full((480, 640), distance / 30.0, np.float32)
                        scale, estimated_tool_depth = calibrator.calibrate_with_tool(
                            relative_depth, mask, intr, ToolDimensionPrior(2.4)
                        )
                        oracle_width_px = intr.fx * 2.4 / tool_depth * np.cos(np.deg2rad(tilt_deg))
                        oracle_tool_depth = intr.fx * 2.4 / max(oracle_width_px, 1e-12)
                        oracle_scale = oracle_tool_depth / float(np.median(relative_depth[mask > 0]))
                        expected_scale = 30.0
                        estimated_lesion_depth = float(np.median(relative_depth[mask > 0]) * scale)
                        oracle_lesion_depth = float(np.median(relative_depth[mask > 0]) * oracle_scale)
                        rows.append({
                            "lesion_depth_mm": distance,
                            "tool_axial_offset_mm": axial_offset,
                            "tool_tilt_deg": tilt_deg,
                            "mask_edge_error_px": mask_edge_error_px,
                            "repeat": repeat,
                            "observed_width_px": width,
                            "estimated_tool_depth_mm": estimated_tool_depth,
                            "oracle_continuous_tool_depth_mm": oracle_tool_depth,
                            "estimated_lesion_depth_mm": estimated_lesion_depth,
                            "oracle_width_lesion_depth_mm": oracle_lesion_depth,
                            "expected_scale_factor": expected_scale,
                            "estimated_scale_factor": scale,
                            "lesion_depth_abs_error_mm": abs(estimated_lesion_depth - distance),
                            "oracle_width_lesion_depth_abs_error_mm": abs(oracle_lesion_depth - distance),
                            "scale_abs_error": abs(scale - expected_scale),
                        })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    conditions = []
    for distance in (20.0, 30.0, 45.0):
        for offset in (0.0, 1.0, 3.0, 5.0):
            for tilt in (0.0, 15.0, 30.0, 45.0):
                subset = [r for r in rows if r["lesion_depth_mm"] == distance and r["tool_axial_offset_mm"] == offset and r["tool_tilt_deg"] == tilt]
                conditions.append({
                    "lesion_depth_mm": distance, "tool_axial_offset_mm": offset, "tool_tilt_deg": tilt,
                    "n": len(subset),
                    "lesion_depth_mae_mm": float(np.mean([r["lesion_depth_abs_error_mm"] for r in subset])),
                    "scale_mae": float(np.mean([r["scale_abs_error"] for r in subset])),
                    "mean_width_px": float(np.mean([r["observed_width_px"] for r in subset])),
                    "oracle_width_lesion_depth_mae_mm": float(np.mean([r["oracle_width_lesion_depth_abs_error_mm"] for r in subset])),
                })
    summary = {
        "seed": SEED,
        "n_trials": len(rows),
        "ideal_condition_mae_by_distance": {
            str(d): float(np.mean([r["lesion_depth_abs_error_mm"] for r in rows if r["lesion_depth_mm"] == d and r["tool_axial_offset_mm"] == 0 and r["tool_tilt_deg"] == 0 and r["mask_edge_error_px"] == 0]))
            for d in (20.0, 30.0, 45.0)
        },
        "conditions": conditions,
        "rows": len(rows),
        "interpretation": "controlled formula test; no clinical accuracy claim",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"n_trials": len(rows), "ideal_mae": summary["ideal_condition_mae_by_distance"]}, indent=2))


if __name__ == "__main__":
    main()
