"""Build the acceptance table from the already generated adaptive OBB test outputs."""
import csv
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "outputs" / "adaptive_obb"
summary = json.loads((out / "adaptive_obb_summary.json").read_text())
trials = list(csv.DictReader((out / "test_trials.csv").open()))
triage = summary["triage_test_rows"]
selected = summary["selected_threshold_deg"]
table = []
for noise in (0, 1, 2, 5, 10, 20):
    subset = [r for r in trials if int(r["noise_pct"]) == noise]
    for method in ("pca_full", "mad_pca", f"adaptive_{selected}"):
        errors = [float(r["mean_abs_extent_error_mm"]) for r in subset if r["method"] == method]
        table.append({
            "noise_pct": noise, "method": method, "n": len(errors),
            "coverage_pct": 100.0, "mae_mm": statistics.mean(errors) if errors else None,
        })
    accepted = [r["mean_abs_extent_error_mm"] for r in triage
                if r["noise_pct"] == noise and r["mean_abs_extent_error_mm"] is not None]
    n_all = sum(r["noise_pct"] == noise for r in triage)
    table.append({
        "noise_pct": noise, "method": "triage_accepted_only", "n": len(accepted),
        "coverage_pct": 100 * len(accepted) / n_all if n_all else 0.0,
        "mae_mm": statistics.mean(accepted) if accepted else None,
    })
(out / "comparison_table.json").write_text(json.dumps(table, indent=2))

triage_metrics = []
for method_name, rows in (("threshold_gate", triage),):
    for noise in (0, 1, 2, 5, 10, 20):
        group = [r for r in rows if r["noise_pct"] == noise]
        accepted = [r for r in group if r["mean_abs_extent_error_mm"] is not None]
        correct_branch = 0
        missed_outlier = 0
        false_alarm = 0
        for r in group:
            should_robust = noise >= 5
            did_robust = r["status"] == "quantify_mad"
            correct_branch += int(should_robust == did_robust) if r["status"] != "abstain_interval" else 0
            missed_outlier += int(should_robust and r["status"] == "quantify_pca")
            false_alarm += int(not should_robust and r["status"] == "quantify_mad")
        triage_metrics.append({
            "method": method_name, "noise_pct": noise, "n": len(group),
            "coverage_pct": 100 * len(accepted) / len(group) if group else 0,
            "selective_mae_mm": float(statistics.mean(r["mean_abs_extent_error_mm"] for r in accepted)) if accepted else None,
            "abstention_pct": 100 * (len(group) - len(accepted)) / len(group) if group else 0,
            "missed_outlier_count": missed_outlier,
            "false_robust_count": false_alarm,
            "branch_accuracy_pct": 100 * correct_branch / len(group) if group else 0,
        })
(out / "triage_metrics.json").write_text(json.dumps(triage_metrics, indent=2))
print(json.dumps({"comparison_table": table, "triage_metrics": triage_metrics}, indent=2))
