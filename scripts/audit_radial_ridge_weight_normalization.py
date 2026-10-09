"""Exploratory audit of scene-weight normalization in the radial calibration study.

IMPORTANT: This is a post-hoc sensitivity analysis on an already observed test set.
It must not be presented as a new confirmatory or untouched test. It preserves the
original frozen study artifacts and writes results into a separate output folder.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))
from c3vdv2_external_depth_validation import metrics, read_reference
from src.radial_affine_calibration import features, predict, radius_squared

OUT = ROOT / "outputs/radial_confirmatory"
AUDIT = ROOT / "outputs/radial_corrected_weight_audit_v2"
GRID = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
FAMILIES = [("affine", False), ("radial", True)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pool(frame_rows: list[dict]) -> dict:
    pool = {}
    for row in frame_rows:
        cache = ROOT / "outputs/publication_study/predictions/development/small" / f"{row['scene']}_{row['frame_id']}.npz"
        with np.load(cache) as z:
            d = z["prediction"].astype(float)
        gt, valid = read_reference(Path(row["depth_path"]))
        idx = np.flatnonzero(valid)
        idx = idx[np.linspace(0, len(idx) - 1, 8192, dtype=int)]
        pool[(row["scene"], row["frame_id"])] = {
            "d": d.ravel()[idx],
            "z": gt.ravel()[idx].astype(float),
            "r2": radius_squared(d.shape).ravel()[idx],
        }
    return pool


def fit_corrected(pool: dict, scenes: list[str], radial: bool, lam: float) -> dict:
    """Fit the manuscript's stated scene-mean objective with normalized weights."""
    scenes = sorted(scenes)
    blocks = []
    for scene in scenes:
        rows = [v for (s, _), v in pool.items() if s == scene]
        merged = {k: np.concatenate([r[k] for r in rows]) for k in ("d", "z", "r2")}
        target_n = 16384 // len(scenes) + int(scenes.index(scene) < 16384 % len(scenes))
        take = np.linspace(0, len(merged["d"]) - 1, target_n, dtype=int)
        blocks.append({k: v[take] for k, v in merged.items()})

    d = np.concatenate([b["d"] for b in blocks])
    z = np.concatenate([b["z"] for b in blocks])
    r2 = np.concatenate([b["r2"] for b in blocks])
    group = np.concatenate([np.full(len(b["d"]), i, dtype=int) for i, b in enumerate(blocks)])
    scene_count = len(scenes)
    counts = np.bincount(group)
    # Correctly normalized scene-balanced objective: sum_i w_i = 1.
    # Each scene has total weight 1 / scene_count.
    weights = 1.0 / (scene_count * counts[group])

    x = features(d, r2, radial)
    mu = np.sum(x * weights[:, None], axis=0)
    scale = np.sqrt(np.sum((x - mu) ** 2 * weights[:, None], axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    design = np.column_stack([(x - mu) / scale, np.ones(len(z))])
    aw = design * np.sqrt(weights[:, None])
    yw = np.sqrt(weights) / z
    penalty = np.diag([1.0] * (design.shape[1] - 1) + [0.0])
    aug = np.vstack([aw, np.sqrt(lam) * penalty])
    target = np.r_[yw, np.zeros(design.shape[1])]
    beta = np.linalg.lstsq(aug, target, rcond=None)[0]
    slopes = beta[:-1] / scale
    intercept = beta[-1] - mu @ slopes
    if radial:
        coeff = [slopes[0], slopes[1], intercept, slopes[2]]
    else:
        coeff = [slopes[0], 0.0, intercept, 0.0]
    return {
        "radial": radial,
        "lambda": lam,
        "feature_mean": mu.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients_a0_a1_b0_b1": list(map(float, coeff)),
        "training_scenes": scenes,
        "weight_sum": float(weights.sum()),
        "objective_scale": "weights sum to 1; equal-scene mean squared inverse-depth error",
    }


def score_scene(head: dict, pool: dict, scene: str) -> float:
    values = []
    for (s, _), a in pool.items():
        if s == scene:
            p = predict(head, a["d"], a["r2"])
            values.append(float(np.abs(p - a["z"]).mean()))
    return float(np.mean(values))


def eval_frame(head: dict, row: dict, cache_path: Path) -> dict:
    with np.load(cache_path) as z:
        d = z["prediction"].astype(float)
    gt, valid = read_reference(Path(row["depth_path"]))
    p = predict(head, d, radius_squared(d.shape))
    return metrics(p, gt, valid)


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((OUT / "protocol.json").read_text())
    frozen = json.loads((OUT / "frozen_heads.json").read_text())
    development_rows = json.loads((ROOT / "outputs/publication_study/development_frames.json").read_text())
    scenes = sorted({r["scene"] for r in development_rows})
    pool = load_pool(development_rows)

    # Lock lambda independently for each ridge family using the same scene-level
    # leave-one-scene-out validation objective; this is post-hoc versus the test.
    selected = {"affine_ls": 0.0, "radial_ls": 0.0}
    cv = []
    for name, radial in FAMILIES:
        if not name in ("affine", "radial"):
            continue
        if name.endswith("_ls"):
            continue
        for lam in GRID:
            losses = []
            for held in scenes:
                head = fit_corrected(pool, [s for s in scenes if s != held], radial, lam)
                loss = score_scene(head, pool, held)
                losses.append(loss)
                cv.append({"family": name, "lambda": lam, "held_scene": held, "mae_mm": loss})
            selected[f"{name}_ridge"] = min(GRID, key=lambda x: np.mean([r["mae_mm"] for r in cv if r["family"] == name and r["lambda"] == x]))

    heads = {}
    for family, radial in FAMILIES:
        for suffix, lam in (("ls", 0.0), ("ridge", selected.get(f"{family}_ridge", 0.0))):
            heads[f"{family}_{suffix}"] = fit_corrected(pool, scenes, radial, lam)

    # Compare frozen official model to corrected objective heads on existing test
    # outputs. These data were already evaluated, so all results are explicitly
    # exploratory sensitivity results and not a confirmatory retest.
    current = json.loads((OUT / "frozen_heads.json").read_text())["heads"]
    test_frames = json.loads((OUT / "test_frames.json").read_text())
    test_rows = []
    for row in test_frames:
        cache = OUT / "predictions" / f"{row['scene']}_{row['frame_id']}.npz"
        with np.load(cache) as z:
            d = z["prediction"].astype(float)
        gt, valid = read_reference(Path(row["depth_path"]))
        for name in ("affine_ls", "affine_ridge", "radial_ls", "radial_ridge"):
            test_rows.append({"scene": row["scene"], "frame": row["frame_id"], "method": name,
                              **eval_frame(heads[name], row, cache)})
        for name in ("affine_ls", "radial_ridge"):
            p = predict(current[name], d, radius_squared(d.shape))
            test_rows.append({"scene": row["scene"], "frame": row["frame_id"], "method": f"original_frozen_{name}",
                              **metrics(p, gt, valid)})

    video_rows = []
    for method in sorted({r["method"] for r in test_rows}):
        for scene in sorted({r["scene"] for r in test_rows}):
            a = [r for r in test_rows if r["method"] == method and r["scene"] == scene]
            if a:
                video_rows.append({"method": method, "scene": scene, "frames": len(a),
                                   **{k: float(np.mean([r[k] for r in a])) for k in
                                      ("mae_mm", "rmse_mm", "abs_rel", "delta1_fraction")}})

    summary = []
    for method in sorted({r["method"] for r in video_rows}):
        videos = [r for r in video_rows if r["method"] == method]
        summary.append({"method": method, "videos": len(videos),
                        **{k: float(np.mean([r[k] for r in videos])) for k in
                           ("mae_mm", "rmse_mm", "abs_rel", "delta1_fraction")}})

    (AUDIT / "normalized_weight_cv.json").write_text(json.dumps({"lambda_grid": GRID, "selected": selected, "records": cv}, indent=2))
    (AUDIT / "normalized_weight_heads.json").write_text(json.dumps(heads, indent=2))
    with (AUDIT / "normalized_weight_test_per_video.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(video_rows[0])); w.writeheader(); w.writerows(video_rows)
    (AUDIT / "normalized_weight_test_summary.json").write_text(json.dumps({
        "status": "post-hoc exploratory audit on already observed test videos; not confirmatory",
        "original_frozen_heads_sha256": sha256(OUT / "frozen_heads.json"),
        "original_protocol_sha256": sha256(OUT / "protocol.json"),
        "corrected_weight_definition": "1 / (number_of_training_scenes * samples_in_scene); weights sum to one",
        "selected_lambda_corrected": selected,
        "summary_equal_video": summary,
        "warning": "Do not substitute these post-hoc results for the frozen confirmatory results. A fresh prospectively frozen test set is required to confirm any change."
    }, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
