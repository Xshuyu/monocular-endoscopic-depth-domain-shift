> 2026-09-26 更新：主论文现在使用10种子、320场景的固定协议结果。详见 [论文实验完善报告](PAPER_STRENGTHENING_REPORT.md)。下文保留的是此前组件/单种子审查记录，不应与新主实验数字混用。

# Current Reverification Summary

This file preserves the earlier component and single-cohort summaries. It reports component-level,
procedural and analytical experiments only; no result below establishes
patient-level clinical size accuracy.

## Reproducibility sources

- Procedural scale/geometry factorial: `tests/verified_reanalysis.py`
- OBB analytic Monte Carlo: `tests/run_independent_verification.py`
- Adaptive OBB validation/test: `tests/adaptive_obb_benchmark.py`
- Tool-scale phantom: `tests/scale_phantom_benchmark.py`
- Curvature held-out validation: `tests/curvature_holdout_validation.py`
- Standard mesh Heat Method validation: `tests/standard_heat_method_validation.py`
- Latest factorial JSON/CSV/manifest: `outputs/verified_reanalysis/`
- Latest OBB Monte Carlo: `outputs/independent_verification/`
- Latest adaptive gate: `outputs/adaptive_obb/`
- Latest phantom and numerical method outputs: `outputs/scale_phantom/`, `outputs/curvature_holdout/`, `outputs/heat_method_validation/`

## Results that can be stated narrowly

### Procedural scale x geometry factorial (N=50)

| Scale mode | PCA full MAE | PCA q01--q99 MAE | MAD-PCA q01--q99 MAE | SO(3) quantile-volume MAE |
|---|---:|---:|---:|---:|
| Oracle depth | 0.665 mm | 0.716 mm | 0.786 mm | 0.855 mm |
| 30-mm prior | 4.163 mm | 4.616 mm | 4.775 mm | 4.850 mm |
| Tool-only | 11.916 mm | 10.617 mm | 10.202 mm | 9.990 mm |
| Equal tool-prior fusion | 3.883 mm | 3.016 mm | 2.732 mm | 2.688 mm |

These are synthetic cohort comparisons. Scale inputs have a large effect. The
tool and prior signed errors are negatively correlated (r≈-0.91), so fusion's
lower average error partly reflects cancellation and is not evidence of robust
calibration. The SO(3)-versus-MAD-PCA difference is small and requires cautious
paired interpretation.

### OBB component experiment

- Clean full-support PCA MAE: approximately 0.041 mm on the analytic ellipsoid.
- SO(3) quantile-volume clean full-support major-axis underestimation: about
  6.67 mm in the fixed 24 x 14 x 6 mm test.
- With explicitly matched radial MAD filtering and endpoint definitions, the
  SO(3) estimator and MAD-PCA have similar outlier errors. This supports the
  filtering step; an independent Lie-group optimization benefit has not been
  demonstrated.
- OBB corner-edge lengths match reported dimensions; rotation perturbation
  changed the tested axis lengths by at most about 0.0088 mm.

### Adaptive OBB gate

Validation-selected rejection thresholds are 6%/8%. On independent test seeds,
clean coverage is 99%; coverage at 5% spatial contamination is only 11.7%, with
accepted-case MAE about 14.24 mm. This is an exploratory abstention candidate,
not a generally accurate estimator.

### Tool-scale phantom

The ideal aligned, same-depth pinhole phantom yields depth MAE approximately
0.034 mm at 20 mm, 1.06 mm at 30 mm and 1.59 mm at 45 mm. Pixel-width
quantization, tilt and axial offset increase error. These are formula/phantom
results and do not validate scale recovery in patient endoscopy.

### Curvature and surface distance

- Curvature on a visible spherical cap is stable over the tested 0--0.01 mm
  normal perturbations after validation-only selection of k=30. The holdout
  sphere errors are about 0.021 mm^-1 for |H| and 0.0033--0.0036 mm^-2 for K.
- Plane and saddle checks give approximately zero curvature and negative
  Gaussian curvature respectively. This does not validate Paris labels.
- Standard cotangent-mesh Heat Method has not passed the predeclared acceptance
  test: interior-source planar max errors are non-monotone and Euclidean chord
  violations remain common. Do not use it in the primary result or claim ESD
  margin validity.

## Dataset truth boundary

Kvasir-SEG has 1000 image/mask pairs with matching IDs and shapes, but no
physical millimetre lesion-size reference. It supports pairing, runtime and
descriptive pipeline checks only—not clinical MAE, CCC or Bland--Altman claims.

## Publication decision

The current evidence supports a limited procedural component study with
explicit failure analysis. It does not support the full clinical MetricEndo
claim set or an external SOTA claim. Before a clinical measurement manuscript,
obtain calibrated phantom or independent post-resection size references and
resolve scale and geometry endpoint mismatches. Curvature/Paris and standard
Heat Method claims should remain excluded until their respective validation
gates pass.
