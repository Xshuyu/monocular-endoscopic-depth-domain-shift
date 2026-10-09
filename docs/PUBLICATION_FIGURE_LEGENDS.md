# Publication Figure Legends

## Figure 1 | Primary endpoint and ablation results

Corrected procedural benchmark on 50 generated scenes with known lesion geometry. Panel a shows major-diameter MAE with deterministic percentile-bootstrap 95% intervals. Panel b shows four-tier risk agreement. The full pipeline and the no-heat ablation both have an MAE of 4.81 mm; the prior-only scale ablation has the lowest MAE (2.59 mm). These results are algorithmic outputs and are not clinical validation.

## Figure 2 | Outlier robustness versus clean-case bias

Ten-seed ellipsoid experiment with 0--20% spatial outlier contamination. Lines show mean MAE and shaded regions show +/-1 s.d. across seeds. PCA has lower clean-case error but becomes unstable under contamination. Robust OBB remains near 3.5--4.0 mm MAE at 5--20% contamination. The figure supports a contamination-robustness claim, not universal unbiased recovery.

## Figure 3 | Analytical curvature validation

Estimated absolute mean curvature and Gaussian curvature for sampled spheres with radii 5, 10 and 20 mm. Dashed lines show analytical values. The radius-10-mm sphere gives approximately H=0.092 mm^-1 and K=0.0092 mm^-2 versus theoretical |H|=0.1 mm^-1 and K=0.01 mm^-2. Curvature recovery is evaluated under controlled sampling and is not a clinical Paris-classification validation.

## Figure 4 | Tool-scale sensitivity

Mean relative scale error under 20, 30 and 45 mm working distances and clean, eroded or dilated tool masks. Values are aggregated over projected-width perturbations from -20% to +20%. The large errors motivate treating the forceps-based scale as an experimental cue rather than a general solution to monocular scale ambiguity.

## Figure 5 | Shape-dependent generalization limits

Major-axis MAE for ellipsoid, dome, plaque and saddle procedural shapes under 0--20% outlier contamination. Panel a shows the uncorrected robust OBB. Panel b shows the exploratory boundary correction selected on a separate validation split. The correction improves ellipsoids but over-corrects several non-ellipsoidal shapes at higher contamination; it is therefore excluded from the primary method.

## Figure 6 | Descriptive real-image assessment

Outputs from the evaluated 50-image Kvasir-SEG subset. Panel a compares Euclidean major-axis and manifold-distance outputs; panel b shows CPU reference latency of 685.95 ms per frame (approximately 1.5 FPS). Kvasir-SEG has no independent physical size or surface-distance reference, so the panel is descriptive and does not establish clinical accuracy or real-time deployment.

## Source and QA

Figures were generated exclusively with Python from:

- `outputs/journal_benchmark_summary.json`
- `outputs/supplementary/supplementary_summary.json`
- `outputs/generalization/summary.json`

Source script: `scripts/plot_publication_figures.py`.

Each figure was exported as editable SVG and PDF plus 600-dpi PNG and TIFF. All six PDFs passed the rendered collision audit and the 5-pt glyph-size audit.
