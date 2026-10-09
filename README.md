# Monocular Endoscopic Depth under Domain Shift: Calibration, Supervised Adaptation and Controlled Failure Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Under%20Review%20(CMIG)-orange.svg)](#)

> **Priority & Manuscript Notice**  
> This repository accompanies the scientific manuscript:  
> **Xiong, S. (2026). *Monocular Endoscopic Depth under Domain Shift: Calibration, Supervised Adaptation and Controlled Failure Analysis*.**  
> **Target Journal:** *Computerized Medical Imaging and Graphics* (Elsevier)  
> **Submission Date:** October 09, 2026  
> **Author & Affiliation:** Shuyu Xiong (Chongqing University of Posts and Telecommunications / Chongqing Open University)  
> **Corresponding Email:** `shuyux@cqou.edu.cn`

---

## 📌 Abstract & Overview

Metric calibration of monocular endoscopic depth often degrades or fails across acquisition domains. In this work, we present a systematic evaluation of four-coefficient radial inverse-depth calibration through matched ablations, independently trained supervised adaptation, and controlled transfer diagnostics across multiple benchmarks (**C3VD**, **SCARED**, **SERV-CT**, and **DRENDS**).

### Key Empirical Findings:
1. **Calibration Transfer Limits**: Frozen affine and radial-ridge heads trained on colon phantom sequences fail when transferred directly to external surgical/laparoscopic cohorts.
2. **True Target Supervision Requirement**: Training a metric decoder on 20 measured SCARED frames reduces held-out SCARED MAE from 36.72 mm to 13.67 mm, and SERV-CT MAE from 55.37 mm to 22.21 mm. Unlabeled alternatives (exposure perturbation, pseudo-label self-training) provide no transfer gain.
3. **Lightweight Closed-form Recalibration**: A two-coefficient target-only recalibration recovers much of the supervised adaptation gains without deep neural network retraining.
4. **Controlled Diagnostics**: Depth-matched factorial contrasts reveal that adapted decoders exhibit depth-dependent blur sensitivity that changes sign across depth bins—a structural behavior obscured by aggregated metrics.

---

## 📂 Repository Structure

```text
├── src/                               # Core Python implementations & mathematical formulations
│   ├── radial_affine_calibration.py   # Radial inverse-depth calibration & ridge regression
│   ├── depth_estimator.py             # Metric depth estimation pipelines
│   ├── robust_metric_depth.py         # Robust metric mapping
│   ├── lie_algebra_obb.py             # Lie algebra oriented bounding box (OBB)
│   ├── standard_heat_method.py        # Heat-method geodesic distance estimation
│   ├── evaluation_suite.py            # Evaluation & metric calculation suites
│   └── camera_model.py                # Pinhole & endoscopic ray models
├── scripts/                           # Reproducible experimental & ablation scripts
│   ├── cmig_revision_common.py        # Frozen protocols, SHA-256 hashes & common utilities
│   ├── run_cmig_calibration.py        # 4-way calibration ablations
│   ├── run_scared_expansion.py        # Supervised adaptation on SCARED splits
│   ├── run_cmig_controls.py           # Source-exposure and pseudo-label control experiments
│   ├── score_cmig_depth_matched.py    # Depth-matched factorial contrast evaluation
│   └── plot_publication_figures.py    # Figure generation scripts (Figures 1-6)
├── tests/                             # Unit tests and validation benchmark suites
├── supplementary/                     # Supplementary review package (S2-S4 artifacts)
│   ├── S2_protocol_environment/       # Frozen protocols, training logs (3 seeds) & SHA-256 logs
│   ├── S3_results/                    # Per-frame and per-object metric summaries (CSV/JSON)
│   ├── S4_figures/                    # High-resolution figures & editable vector graphics (SVG/DrawIO)
│   └── SUPPLEMENTARY_FILE_INDEX.csv   # Itemized index and description of all supplementary artifacts
├── docs/                              # Experimental protocols and reproduction audits
├── INITIAL_COMMIT.md                  # Priority and submission statement
├── LICENSE                            # MIT License
└── setup.py                           # Package installation configuration
```

---

## 🔬 Reviewer Reproduction Guide

To facilitate transparent and rigorous peer review, this repository provides two reproduction pathways:
1. **Lightweight Numerical & Figure Reproduction (No dataset download required)**: Inspect frozen training logs, per-frame metric CSVs, and regenerate paper figures/tables directly from `supplementary/`.
2. **Full End-to-End Pipeline Reproduction**: Train the decoders from scratch, re-run inference across datasets, and recalculate metrics.

### 1. Environment Installation

```bash
git clone git@github.com:Xshuyu/monocular-endoscopic-depth-domain-shift.git
cd monocular-endoscopic-depth-domain-shift

conda create -n endo_depth python=3.10 -y
conda activate endo_depth

# Install dependencies and register editable package
pip install -e .
```

Verify core algorithms and unit test suite:
```bash
python -m unittest tests/test_radial_affine_calibration.py tests/test_radial_measurement_integration.py
```

---

### 2. Fast Reproduction from Supplementary Artifacts

Reviewers can verify the exact numerical metrics and audit logs reported in the manuscript without downloading multi-gigabyte video datasets:

- **Audit Training Curves & Checkpoint Hashes**:
  Inspect `supplementary/S2_protocol_environment/` for training convergence logs across all 3 random seeds (`11`, `23`, `47`) for `source_half`, `target_only`, and `adapted_double`.
- **Inspect Per-frame / Per-object Metrics**:
  Explore `supplementary/S3_results/` for granular CSV records matching Tables 1–4.
- **Inspect Figures & Vector Sources**:
  View `supplementary/S4_figures/` for high-resolution vector PDFs and source SVGs (`radial_architecture.svg`, `scared_subject_results.svg`).

---

### 3. End-to-End Experimental Reproduction

To re-run the complete training and evaluation pipelines from raw data:

#### A. Data Preparation
Organize downloaded datasets under `datasets/` conforming to the paths specified in `scripts/cmig_revision_common.py`:
```text
datasets/
├── external_models/
│   └── Depth-Anything-V2-Small-hf/   # Pretrained generic backbone weights
└── external/
    ├── C3VD/                          # C3VD colon phantom sequences
    ├── SCARED_official/              # SCARED endoscopic dataset with keyframe depth
    ├── SERV-CT/                       # SERV-CT surgical stereo benchmark
    └── DRENDS/                        # DRENDS ex-vivo / phantom validation set
```

#### B. Execution Commands

1. **Four-way Calibration Ablation (Table 1 & Table 2)**:
   ```bash
   python scripts/run_cmig_calibration.py
   ```
2. **Supervised Adaptation & Control Regimes (3 seeds)**:
   ```bash
   python scripts/run_cmig_controls.py
   ```
3. **Pseudo-label Self-training Control**:
   ```bash
   python scripts/run_cmig_selftrain.py
   ```
4. **Depth-matched Factorial Diagnostics (Figure 5 & Figure 6)**:
   ```bash
   python scripts/score_cmig_depth_matched.py
   ```
5. **Generate Publication Figures**:
   ```bash
   python scripts/plot_publication_figures.py
   ```

---

## 🔒 Data & Pretrained Weights Availability Policy

In compliance with data usage terms of the original consortiums:
- **Datasets**: Raw video sequences and patient CT references (C3VD, SCARED, SERV-CT, DRENDS) are subject to third-party agreements and cannot be redistributed directly. Researchers should request access via the official sources cited in the manuscript.
- **Checkpoints**: Pretrained foundation models follow their original upstream licenses.

---

## 📑 Citation

```bibtex
@article{xiong2026monocular,
  title={Monocular Endoscopic Depth under Domain Shift: Calibration, Supervised Adaptation and Controlled Failure Analysis},
  author={Xiong, Shuyu},
  journal={Computerized Medical Imaging and Graphics},
  year={2026},
  note={Submitted on October 09, 2026}
}
```

---

## 📜 License

This codebase is distributed under the [MIT License](LICENSE).
