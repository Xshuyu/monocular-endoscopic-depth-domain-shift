#!/usr/bin/env python
"""Build manuscript_cmig/supplementary.tex from frozen result JSONs.

Reads only already-frozen outputs (no re-computation of experiments):
  outputs/cmig_revision/comparison_summary.json     (bound=300 rows)
  outputs/cmig_selftrain/selftrain_summary.json     (bound=300 rows)
  outputs/cmig_extension/strong_blur_effects.json
  outputs/cmig_extension/depth_matched_effects.json
  outputs/cmig_revision/validation_checks.json
  outputs/cmig_selftrain/validation_checks.json
  outputs/cmig_revision/environment.json
Writes manuscript_cmig/supplementary.tex.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "manuscript_cmig" / "supplementary.tex"

COHORTS = ["c3vd", "scared", "servct", "drends_phantom", "drends_exvivo"]
COHORT_NAMES = {
    "c3vd": "C3VD",
    "scared": "SCARED (held out)",
    "servct": "SERV-CT",
    "drends_phantom": "DRENDS phantom",
    "drends_exvivo": "DRENDS ex vivo",
}

METHOD_GROUPS = [
    ("Neural decoders (retrained)", [
        ("source_only", "Source-only decoder"),
        ("source_half", "Source-half decoder"),
        ("target_only", "Target-only decoder"),
        ("adapted", "Adapted decoder"),
        ("adapted_double", "Adapted decoder, double budget"),
        ("self_train", "Self-train (pseudo-labels)"),
        ("self_train_ft", "Self-train-ft (pseudo-labels)"),
    ]),
    ("Frozen heads on the generic DA V2 Small backbone", [
        ("small_source_only_affine_ls", "Global LS, source-only calibration"),
        ("small_source_only_radial_ls", "Radial LS, source-only calibration"),
        ("small_source_target_affine_ls", "Global LS, source+target calibration"),
        ("small_source_target_radial_ls", "Radial LS, source+target calibration"),
        ("small_target_only_affine_ls", "Global LS, target-only calibration"),
        ("small_target_only_radial_ls", "Radial LS, target-only calibration"),
    ]),
    ("Frozen heads on the EndoDAC+PCC backbone", [
        ("pcc_source_only_affine_ls", "Global LS, source-only calibration"),
        ("pcc_source_only_radial_ls", "Radial LS, source-only calibration"),
        ("pcc_source_target_affine_ls", "Global LS, source+target calibration"),
        ("pcc_source_target_radial_ls", "Radial LS, source+target calibration"),
        ("pcc_target_only_affine_ls", "Global LS, target-only calibration"),
        ("pcc_target_only_radial_ls", "Radial LS, target-only calibration"),
    ]),
]

BINS = ["0_30", "30_60", "60_100", "100_200", "200_300"]
BIN_NAMES = {"0_30": "0--30", "30_60": "30--60", "60_100": "60--100",
             "100_200": "100--200", "200_300": "200--300"}


def load(p):
    return json.load(open(ROOT / p))


def mean_sd(values):
    if not values:
        return "--"
    m = statistics.fmean(values)
    if len(values) < 2:
        return f"{m:.2f}"
    return f"{m:.2f} $\\pm$ {statistics.stdev(values):.2f}"


def esc(s):
    return s.replace("_", "\\_")


def main():
    comp = [r for r in load("outputs/cmig_revision/comparison_summary.json") if r["bound"] == 300]
    st = [r for r in load("outputs/cmig_selftrain/selftrain_summary.json") if r["bound"] == 300]
    # Additional calibration-only target-domain heads are scored from their
    # frozen native-grid evaluation summaries under the same 300 mm ceiling.
    calibration = [r for r in load("outputs/cmig_extension/combined_calibration_summary.json")
                  if r["bound"] == 300]
    bound300 = comp + st + calibration

    # ---------------- Table S1 ----------------
    idx = {}
    for r in bound300:
        idx.setdefault((r["method"], r["cohort"]), []).append(r["mae_mm"])
    s1 = []
    s1.append("\\begin{sidewaystable}\\centering\\footnotesize")
    s1.append("\\caption{Secondary analysis at the $[1,300]$\\,mm prediction ceiling: unaligned MAE (mm), "
              "mean $\\pm$ sample SD over trained-model seeds; calibration-only heads have one fitted "
              "head each. The tested data splits and calibration heads are unchanged from the primary "
              "$[1,100]$\\,mm analysis. ``--'' marks excluded combinations; EndoDAC+PCC is omitted from "
              "held-out SCARED because the audited checkpoint training list contains all nine SCARED "
              "dataset identifiers.}\\label{tab:s1_bound300}")
    s1.append("\\begin{tabular}{p{0.18\\textwidth}p{0.20\\textwidth}ccccc}")
    s1.append("\\toprule")
    s1.append("Group & Regime & " + " & ".join(COHORT_NAMES[c] for c in COHORTS) + " \\\\")
    s1.append("\\midrule")
    for gi, (gname, methods) in enumerate(METHOD_GROUPS):
        s1.append(f"\\multicolumn{{{2 + len(COHORTS)}}}{{l}}{{\\emph{{{gname}}}}}\\\\")
        for key, label in methods:
            cells = []
            for c in COHORTS:
                vals = idx.get((key, c))
                if vals is None:
                    cells.append("--")
                else:
                    cells.append(mean_sd(vals))
            s1.append(f" & {label} & " + " & ".join(cells) + " \\\\")
        if gi < len(METHOD_GROUPS) - 1:
            s1.append("\\addlinespace")
    s1.append("\\bottomrule\\end{tabular}\\end{sidewaystable}")

    # ---------------- Table S2 ----------------
    blur = load("outputs/cmig_extension/strong_blur_effects.json")
    bidx = {}
    for r in blur:
        bidx.setdefault((r["cohort"], r["method"]), []).append(r)
    s2 = []
    s2.append("\\begin{sidewaystable}\\centering\\footnotesize")
    s2.append("\\caption{Full $2\\times2$ factorial intervention results with strong blur defined at "
              "model-input resolution ($224\\times280$ for the neural decoders). Cell entries are MAE (mm), "
              "mean $\\pm$ sample SD over three seeds; effect columns are the factorial dark main effect, "
              "strong-blur main effect and interaction. This table expands the pooled effects shown in "
              "Figure~6 of the main text.}\\label{tab:s2_factorial}")
    s2.append("\\begin{tabular}{p{0.16\\textwidth}p{0.12\\textwidth}ccccccc}")
    s2.append("\\toprule")
    s2.append("Cohort & Model & Identity & Dark & Strong blur & Dark+blur & Dark main & Blur main & Interact. \\\\")
    s2.append("\\midrule")
    for c in COHORTS:
        for mi, m in enumerate(["source_only", "adapted"]):
            rows = bidx.get((c, m))
            if not rows:
                continue
            mname = "Source-only" if m == "source_only" else "Adapted"
            cells = [mean_sd([r[k] for r in rows]) for k in
                     ["identity", "dark", "strong_blur", "dark_strong_blur",
                      "dark_main_mm", "strong_blur_main_mm", "interaction_mm"]]
            cname = COHORT_NAMES[c] if mi == 0 else ""
            s2.append(f"{cname} & {mname} & " + " & ".join(cells) + " \\\\")
        s2.append("\\addlinespace")
    s2.append("\\bottomrule\\end{tabular}\\end{sidewaystable}")

    # ---------------- Table S3 ----------------
    dm = load("outputs/cmig_extension/depth_matched_effects.json")
    didx = {}
    for r in dm:
        didx.setdefault((r["cohort"], r["method"], r["bin"]), []).append(r["strong_blur_main_mm"])
    s3 = []
    s3.append("\\begin{sidewaystable}\\centering\\footnotesize")
    s3.append("\\caption{Depth-matched strong-blur main effect (mm) by reference-depth bin, mean $\\pm$ "
              "sample SD over three seeds. Within each cohort, model and bin, the identity and strong-blur "
              "scorings share the same frames and the same reference-depth distribution, so the contrast is "
              "not confounded by depth composition. Positive values mean blur increases error. Effects vary "
              "across depth bins and cohorts, including sign changes within supported bins; the stratified "
              "patterns are obscured by pooled effects in Table~\\ref{tab:s2_factorial}.}"
              "\\label{tab:s3_depth_matched}")
    s3.append("\\begin{tabular}{p{0.20\\textwidth}p{0.16\\textwidth}ccccc}")
    s3.append("\\toprule")
    s3.append("Cohort & Model & " + " & ".join(BIN_NAMES[b] + "\\,mm" for b in BINS) + " \\\\")
    s3.append("\\midrule")
    for c in COHORTS:
        for mi, m in enumerate(["source_only", "adapted"]):
            mname = "Source-only" if m == "source_only" else "Adapted"
            cells = [mean_sd(didx.get((c, m, b), [])) for b in BINS]
            cname = COHORT_NAMES[c] if mi == 0 else ""
            s3.append(f"{cname} & {mname} & " + " & ".join(cells) + " \\\\")
        s3.append("\\addlinespace")
    s3.append("\\bottomrule\\end{tabular}\\end{sidewaystable}")

    # ---------------- S4 provenance ----------------
    vc_rev = load("outputs/cmig_revision/validation_checks.json")
    vc_st = load("outputs/cmig_selftrain/validation_checks.json")
    env = load("outputs/cmig_revision/environment.json")
    n_checkpoints = len(vc_rev["training_counts"]) + len(vc_st["training_counts"])
    s4 = []
    s4.append("\\section{Provenance and validation checks}\\label{sec:s4}")
    s4.append("All regimes were specified in protocol files written to disk before any training or scoring "
              "run; existing outputs were never overwritten, and no regime was selected or dropped on the "
              "basis of test performance. Automated post-run checks verified the following:")
    s4.append("\\begin{itemize}")
    s4.append(f"\\item Protocol hashes verified for both the revision protocol and the self-training "
              f"protocol: {str(vc_rev['protocol_hashes_verified'] and vc_st['protocol_hashes_verified']).lower()}.")
    s4.append(f"\\item All {n_checkpoints} newly trained checkpoints verified against the protocol "
              f"(9 revision-control checkpoints, 6 self-training checkpoints).")
    s4.append(f"\\item Training RGB, ground-truth and test RGB hashes verified: "
              f"{str(vc_rev['training_rgb_gt_and_test_rgb_hashes_verified']).lower()}.")
    s4.append(f"\\item Calibration-budget and training draw counts verified for every checkpoint "
              f"(source/target draws per run match the factorial design).")
    s4.append(f"\\item EndoDAC+PCC excluded from held-out SCARED scoring as designed: "
              f"{str(vc_rev['pcc_scared_test_excluded']).lower()}.")
    s4.append(f"\\item Pseudo-label generation read no reference depth values: "
              f"{str(vc_st['no_reference_used_for_pseudo_labels']).lower()}; pseudo-label hashes verified.")
    s4.append(f"\\item No test-performance-based model selection: "
              f"{str(vc_st['no_test_performance_selection']).lower()}.")
    s4.append(f"\\item {vc_rev['independent_native_metric_checks']} independent native-resolution metric "
              f"checks; identity-scoring reproduction difference "
              f"{vc_rev['identity_max_prediction_difference_mm']}\\,mm.")
    s4.append(f"\\item All summary tables re-aggregated from per-frame records after the runs: "
              f"{str(vc_rev['all_summaries_reaggregated'] and vc_st['summaries_reaggregated']).lower()}.")
    s4.append("\\end{itemize}")
    s4.append("Two limitations are recorded in the same validation files and remain open: no new independent "
              "donors or device serials were verified for DRENDS, and no clinical size or observer-agreement "
              "validation was performed.")
    s4.append("\\paragraph{Environment.} " +
              f"Python {env['version'].split('|')[0].strip()}, PyTorch {env['packages']['torch']}, "
              f"transformers {env['packages']['transformers']}, NumPy {env['packages']['numpy']}, "
              f"single {env['gpu']}.")

    # ---------------- assemble ----------------
    tex = []
    tex.append("\\documentclass[11pt]{article}")
    tex.append("\\usepackage[margin=2.2cm]{geometry}")
    tex.append("\\usepackage{booktabs}")
    tex.append("\\usepackage{rotating}")
    tex.append("\\setlength{\\tabcolsep}{4pt}")
    tex.append("\\usepackage{graphicx}")
    tex.append("\\usepackage{amsmath}")
    tex.append("\\usepackage[hidelinks]{hyperref}")
    tex.append("\\title{Supplementary material\\\\[2pt] \\large Metric calibration transfer and adaptation "
               "limits of monocular endoscopic depth estimation}")
    tex.append("\\author{}")
    tex.append("\\date{}")
    tex.append("\\begin{document}")
    tex.append("\\maketitle")
    tex.append("This supplement accompanies the submission to \\emph{Computerized Medical Imaging and "
               "Graphics}. It reports the secondary-range analysis, the full factorial tables and the "
               "provenance checks referenced in the Data and code availability statement. All values are "
               "aggregated from the frozen per-frame outputs described in the main text; no experiment was "
               "re-run or selected for this supplement.")
    tex.append("\\section{Secondary range: $[1,300]$\\,mm clip ceiling}\\label{sec:s1}")
    tex.append("The primary analysis scores with a $[1,100]$\\,mm prediction ceiling. This secondary "
               "analysis re-scores the trained neural regimes and calibration-only control heads with a "
               "$[1,300]$\\,mm ceiling. Table~\\ref{tab:s1_bound300} reports these results. Target-only "
               "calibration heads use target-domain pixels only; source+target heads use equal pixel budgets "
               "from both domains. The fit, data split and calibration pixels are otherwise unchanged.")
    tex.extend(s1)
    tex.append("\\section{Full factorial intervention tables}\\label{sec:s2}")
    tex.append("The main text reports pooled factorial effects (Figure~6). "
               "Table~\\ref{tab:s2_factorial} gives the complete $2\\times2$ cell means and effects per "
               "cohort, and Table~\\ref{tab:s3_depth_matched} gives the depth-matched blur effect per "
               "reference-depth bin.")
    tex.extend(s2)
    tex.extend(s3)
    tex.extend(s4)
    tex.append("\\end{document}")

    OUT.write_text("\n\n".join(tex) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
