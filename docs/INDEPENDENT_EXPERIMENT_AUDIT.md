# Independent Experiment Audit — Current State

## Source of truth

- Procedural factor analysis: `outputs/verified_reanalysis/verified_reanalysis.json`
- Per-case predictions: `outputs/verified_reanalysis/procedural_factorial_trials.csv`
- Input/output hashes: `outputs/verified_reanalysis/manifest.json`
- OBB component trials: `outputs/independent_verification/verification_summary.json` and `obb_trials.csv`
- Adaptive OBB validation/test: `outputs/adaptive_obb/adaptive_obb_summary.json`, `validation_trials.csv`, `test_trials.csv`, `comparison_table.json`, `triage_metrics.json`
- Scale formula phantom: `outputs/scale_phantom/summary.json` and `trials.csv`
- Standard mesh Heat Method test: `outputs/heat_method_validation/summary.json` and `heat_method_convergence.csv`
- Publication figure source outputs: `outputs/verified_figures/`

Legacy `verified/*.json`, `journal_benchmark_summary.json`, and root-level figures are historical only and must not be mixed into current results.

## Code corrections made

1. OBB SO(3) updates are now applied once per iteration and use monotone backtracking. The algorithm is accurately described as MAD-trimmed quantile-volume OBB; Huber loss is not part of this optimizer.
2. Removed blanket 95th-percentile radial trimming before all geometry methods.
3. Fixed output corner axis ordering so reported extents equal the corner edge lengths.
4. Added explicit `never/always/adaptive` radial-MAD filtering modes and matched MAD-PCA controls.
5. Removed unjustified scale-factor clipping from the reanalysis. Oracle, prior, tool-only and fusion modes are separate factors.
6. Made curvature sampling deterministic and made normal orientation an explicit option.
7. Reclassified the legacy point-cloud distance as a heat-kernel/kNN hybrid, not the standard Heat Method. Added a separate standard triangle-mesh implementation and convergence test; it has not passed acceptance.

## Verified comparisons

The procedural factorial experiment uses 50 paired scenes and 800 predictions (4 scale conditions × 4 geometry estimators). Current major-diameter MAE (mm):

| Scale | PCA full | PCA 1--99% | MAD-PCA | SO(3) quantile-volume |
|---|---:|---:|---:|---:|
| Oracle depth | 0.665 | 0.716 | 0.786 | 0.855 |
| 30-mm prior | 4.163 | 4.616 | 4.775 | 4.850 |
| Tool-only | 11.916 | 10.617 | 10.202 | 9.990 |
| 50/50 fusion | 3.883 | 3.016 | 2.732 | 2.688 |

Bootstrap 95% intervals, paired differences, Wilcoxon tests and Holm-adjusted p-values are in the JSON. The SO(3) gain over MAD-PCA under fusion is small. Tool and prior signed errors correlate at approximately -0.91, showing that fusion partly benefits from cancellation of opposing errors; do not describe it as validated robust calibration.

The separate analytic OBB component benchmark shows that clean PCA full-support is highly accurate, while SO(3) quantile-volume underestimates the complete clean ellipsoid major axis. At contamination, MAD filtering materially stabilizes both MAD-PCA and SO(3), and their errors are similar. This supports MAD filtering as a robustness step, not a demonstrated independent Lie-group advantage.

The adaptive rejection gate selects 6%/8% thresholds on validation seeds. Test clean coverage is 99%; 5% contamination coverage is only 11.7%, and accepted-case MAE is still about 14.24 mm. It remains an exploratory abstention strategy, not an accepted accuracy improvement.

## Component acceptance status

| Component | Status | Evidence boundary |
|---|---|---|
| Camera project/back-project | Passes algebraic round-trip | Does not validate camera intrinsics for a real endoscope |
| OBB corner geometry | Passes reported-size/corner-edge consistency; rotation perturbation change <0.01 mm in the tested case | Clean full-support bias remains; no independent SO(3) advantage established over matched MAD-PCA |
| Tool scale formula | Formula executes on known geometry | Strongly sensitive to pixel quantization, mask, tilt and axial offset; not clinical metric calibration |
| Curvature | Deterministic sampling and analytic visible-cap tests run | Gaussian curvature fails noise robustness; Paris classification is not validated |
| Standard Heat Method | Mesh implementation and analytic convergence tests exist | Acceptance not passed: interior-source plane maxima and geodesic-vs-chord lower-bound violations are not stable across tested refinements |
| Kvasir data pairing | 1000 images, 1000 masks, 1000 matching IDs, zero shape mismatches | No millimetre-level ground truth; no clinical size-accuracy result |

The controlled scale phantom comprises 7,200 trials. With continuous projected-width oracle, same-depth and axis-aligned conditions recover the scale equation to numerical precision. Rasterized `minAreaRect` width produces lesion-depth MAE approximately 0.034 mm at 20 mm, 1.06 mm at 30 mm and 1.59 mm at 45 mm. A 3-mm tool/lesion axial offset yields about 3 mm ideal-width error; a 30-degree tilt yields about 4.64 mm; combined 5-mm offset and 45-degree tilt produces much larger error. These results demonstrate sensitivity, not clinical accuracy. Results rerun with current `scale_phantom_benchmark.py` on the recorded seed.

## Decisions for any manuscript

- Do not claim clinical metric accuracy, in-vivo caliper agreement, validated Paris classification, ESD margin accuracy, or SOTA superiority.
- If writing a manuscript now, restrict claims to procedural component evaluation and failure analysis.
- Exclude curvature-derived clinical labels and the unpassed standard Heat Method from primary claims.
- Keep Kvasir-SEG results descriptive (pair integrity/runtime/output distribution only).
- Any further code or protocol change requires rerunning the named source script, regenerating the manifest and figures, and updating this document.
