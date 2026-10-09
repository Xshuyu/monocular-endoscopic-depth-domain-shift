# Monocular Endoscopic Depth under Domain Shift: Calibration, Supervised Adaptation and Controlled Failure Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Under%20Review%20(CMIG)-orange.svg)](#)

> **Priority & Manuscript Notice**  
> This repository accompanies the scientific manuscript:  
> **Xiong, S. (2026). *Monocular Endoscopic Depth under Domain Shift: Calibration, Supervised Adaptation and Controlled Failure Analysis*.**  
> **Submitted to:** *Computerized Medical Imaging and Graphics* (Elsevier)  
> **Submission Date:** October 09, 2026  
> **Corresponding Author:** Shuyu Xiong (Chongqing University of Posts and Telecommunications / Chongqing Open University)

---

## 📌 Overview & Abstract

Metric calibration of monocular endoscopic depth often degrades or fails across different acquisition setups. In this work, we present a systematic evaluation of four-coefficient radial inverse-depth calibration through matched ablations, independently trained supervised adaptation, and controlled transfer diagnostics across multiple benchmarks (**C3VD**, **SCARED**, **SERV-CT**, and **DRENDS**).

### Key Empirical Findings:
1. **Transfer Limits of Spatial Calibration**: Frozen affine and radial-ridge heads trained on colon phantom sequences fail when transferred directly to external surgical/laparoscopic cohorts.
2. **True Target Supervision Requirement**: Training a metric decoder on 20 measured SCARED frames reduces held-out SCARED MAE from 36.72 mm to 13.67 mm, and SERV-CT MAE from 55.37 mm to 22.21 mm. Unlabeled controls (source exposure perturbation, pseudo-label self-training) provide no transfer gain.
3. **Closed-form Two-coefficient Recalibration**: A two-coefficient target-only recalibration recovers much of the supervised adaptation gains without deep neural network retraining.
4. **Controlled Diagnostics**: Depth-matched factorial contrasts reveal that adapted decoders exhibit depth-dependent blur sensitivity that changes sign across depth bins—a structural behavior obscured by aggregated metrics.

---

## 📂 Repository Structure

```text
├── src/                               # Core Python implementations
│   ├── radial_affine_calibration.py   # Radial inverse-depth calibration & ridge regression
│   ├── depth_estimator.py             # Metric depth estimation pipelines
│   ├── robust_metric_depth.py         # Robust metric mapping
│   ├── lie_algebra_obb.py             # Lie algebra oriented bounding box (OBB)
│   ├── standard_heat_method.py        # Heat-method geodesic distance estimation
│   ├── evaluation_suite.py            # Evaluation & metric calculation suites
│   └── camera_model.py                # Pinhole & endoscopic ray models
├── scripts/                           # Reproducible experimental & ablation scripts
│   ├── run_cmig_calibration.py        # 4-way calibration ablations
│   ├── run_scared_expansion.py        # Supervised adaptation on SCARED splits
│   ├── run_cmig_controls.py           # Source-exposure and pseudo-label control experiments
│   ├── score_cmig_depth_matched.py    # Depth-matched factorial contrast evaluation
│   └── plot_publication_figures.py    # Figure generation scripts
├── tests/                             # Unit tests and validation benchmark suites
├── docs/                              # Experimental protocols and reproduction audits
├── INITIAL_COMMIT.md                  # Priority and submission statement
├── LICENSE                            # MIT License
└── README.md                          # Repository documentation
```

---

## 🚀 Environment & Quick Start

### 1. Requirements

```bash
git clone git@github.com:Xshuyu/monocular-endoscopic-depth-domain-shift.git
cd monocular-endoscopic-depth-domain-shift

conda create -n endo_depth python=3.10 -y
conda activate endo_depth
pip install torch torchvision numpy scipy scikit-learn matplotlib
```

### 2. Running Calibration Ablations

```bash
python scripts/run_cmig_calibration.py
```

### 3. Evaluating Controlled Diagnostic Contrasts

```bash
python scripts/score_cmig_depth_matched.py
```

---

## 🔒 Data & Checkpoint Availability Policy

- **Datasets**: In accordance with the license conditions of C3VD, SCARED, SERV-CT, and DRENDS, raw endoscopic datasets and patient CT references are not redistributed in this repository. Please obtain datasets directly from the respective consortiums.
- **Weights**: Model weights for foundation backbones follow their respective upstream repositories.

---

## 📑 Citation

If you find this work or the codebase helpful in your research, please cite:

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
