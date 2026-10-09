# Adaptive OBB remediation experiment

Script: `tests/adaptive_obb_benchmark.py`  
Summary: `outputs/adaptive_obb/adaptive_obb_summary.json`  
Comparison table: `outputs/adaptive_obb/comparison_table.json`  
Selective metrics: `outputs/adaptive_obb/triage_metrics.json`

## Candidate method

The gate measures the fraction of points rejected by the radial MAD rule. If
the rejection fraction is below a low threshold, it reports PCA full-support
size. Above a high threshold it reports MAD-filtered robust OBB size. In the
intermediate range it abstains from a single value. Thresholds were selected
on validation seeds 100--199 and evaluated on disjoint test seeds 200--499.

The validation protocol required clean MAE non-inferiority within 0.25 mm and
at least 95% clean coverage. The selected rejection-fraction thresholds were
6% and 8%.

This score is not a validated clinical artifact detector: its distributions
overlap substantially across clean, 1% and 2% contamination conditions. The
selected thresholds therefore abstain on many moderate-noise cases.

## Test-set coverage and selective error

| Outlier contamination | Coverage | Abstention | Selective MAE (mm) | Missed robust-branch cases |
|---:|---:|---:|---:|---:|
| 0% | 99.0% | 1.0% | 0.18 | 0/300 |
| 1% | 91.7% | 8.3% | 56.28 | 0/300 |
| 2% | 76.3% | 23.7% | 67.53 | 0/300 |
| 5% | 11.7% | 88.3% | 14.24 | 3/300 |
| 10% | 100.0% | 0.0% | 6.60 | 0/300 |
| 20% | 100.0% | 0.0% | 9.80 | 0/300 |

The method protects clean-case output and abstains frequently in ambiguous
conditions, but at 5% contamination its coverage is only 11.7% and accepted
case error remains high. At 1--2%, the gate often stays on PCA despite large
error because the rejection statistic overlaps with clean cases. This candidate
does not meet a clinically useful accuracy/coverage criterion and is not
accepted as a general performance improvement.

## Decision

Keep the gate as a simulation-only research direction. Do not describe it as
SOTA, clinical sizing, or a validated confidence estimator. Real artifact
masks, depth uncertainty, and calibrated phantom data are needed to learn or
validate a useful operating threshold.
