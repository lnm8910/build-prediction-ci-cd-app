# Temporal Data Leakage Prevention in CI/CD Build Prediction
## Replication Package for PLOS ONE Submission

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Complete replication package for the paper:

**"A taxonomy for detecting and preventing temporal data leakage in machine learning-based build prediction: A dual-platform empirical validation"**

📄 Journal: PLOS ONE
👥 Authors: Amit Rangari¹, Lalit Narayan Mishra²*, Sandesh Nagrare³, Saroj Kumar Nayak⁴

¹ JPMorgan Chase & Co, Atlanta, GA, USA
² Lowe's Companies, Inc., Charlotte, NC, USA (Corresponding Author)
³ Vidyalankar Institute of Technology, Mumbai, India
⁴ Amrita School of Computing, Amrita Vishwa Vidyapeetham, India

---

## Overview

This repository contains the complete implementation, data, and documentation for our dual-platform study on preventing temporal data leakage in CI/CD build prediction. We analyze **175,706 builds** across two independent datasets spanning **11 years** (TravisTorrent 2013-2017, GHALogs 2023).

### Key Contribution

**3-Type Temporal Data Leakage Taxonomy:**
- **Type 1: Direct Outcome Encoding** - Features explicitly encoding build results
- **Type 2: Execution-Dependent Metrics** - Features computable only after build execution
- **Type 3: Future Information Leakage** - Features incorporating data from after prediction time

**Result:** Systematic filtering from 66/33 raw features → **31/29 clean pre-build features**

### Main Results

#### TravisTorrent (Travis CI, 2013-2017)
- ✅ **82.73%** accuracy with only pre-build features
- ✅ **91.38%** ROC-AUC (strong discrimination)
- ✅ **15.07pp** leakage tax (severe inflation from time-dependent features)

#### GHALogs (GitHub Actions, 2023)
- ✅ **83.30%** accuracy with only pre-build features
- ✅ **80.10%** ROC-AUC
- ✅ **0.48pp** leakage tax (minimal inflation from time-dependent features)

#### Platform-Dependent Discovery
- **14.59 percentage point divergence** in leakage tax between platforms
- Travis CI (3rd party service): Substantial time-dependent leakage
- GitHub Actions (native integration): Minimal time-dependent leakage
- **Novel finding:** Platform architecture fundamentally affects metadata predictiveness

---

## Repository Structure

```
plos-one-build-prediction-replication/
├── data/                    # Processed datasets
│   ├── travistorrent/       # 100K builds, 31 clean features
│   └── ghalogs/             # 75.7K workflows, 29 clean features
│
├── src/                     # Source code
│   ├── leakage_detection/   # ⭐ Core toolkit (3-type taxonomy)
│   ├── preprocessing/       # Data preprocessing for both platforms
│   ├── models/              # ML model training (RF, GB, LR)
│   ├── evaluation/          # Statistical tests and metrics
│   └── visualization/       # Figure generation (Figures 1-5)
│
├── experiments/             # Reproducible experiments
│   ├── travistorrent/       # RQ1-RQ4 experiments
│   ├── ghalogs/             # RQ1, RQ5 experiments
│   └── combined/            # Cross-platform analysis
│
├── results/                 # Pre-computed results
│   ├── travistorrent/       # Tables 1-5, Figures 1-3
│   └── ghalogs/             # Table 6, Figures 4-5
│
└── docs/                    # Comprehensive documentation
    ├── INSTALLATION.md      # Setup instructions
    ├── USAGE_GUIDE.md       # How to run experiments
    ├── METHODOLOGY.md       # Detailed methodology
    ├── LEAKAGE_TAXONOMY.md  # 3-type taxonomy explained
    └── FEATURE_MAPPINGS.md  # Platform-specific mappings
```

---

## Quick Start

### Installation

```bash
# 1. Clone repository
git clone https://github.com/lnm8910/plos-one-build-prediction.git
cd plos-one-build-prediction

# 2. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR: venv\Scripts\activate  # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python test_installation.py
```

**Expected output:** All 6 tests should pass (imports, data loading, toolkit, model training)

### Download Datasets

The processed datasets are included in this package. Raw datasets available at:
- **TravisTorrent:** https://travistorrent.testroots.org/ (DOI: 10.5281/zenodo.1254890)
- **GHALogs:** https://zenodo.org/record/10154920 (DOI: 10.5281/zenodo.10154920)

**Note:** The included GHALogs dataset contains 85,574 workflow runs (full enriched version). The paper analysis used a filtered subset of 75,706 runs. Both samples exceed statistical power requirements and yield statistically equivalent results (difference <0.5%)

### Run Experiments

```bash
# Reproduce all TravisTorrent results (RQ1-RQ4)
cd experiments/travistorrent
./run_all_travistorrent.sh

# Reproduce all GHALogs results (RQ1, RQ5)
cd experiments/ghalogs
./run_all_ghalogs.sh

# Run complete cross-platform analysis
cd experiments/combined
python platform_comparison.py
```

---

## Key Features

### 1. Automated Leakage Detection Toolkit

```python
from src.leakage_detection import TemporalLeakageTaxonomy

# Initialize taxonomy
taxonomy = TemporalLeakageTaxonomy()

# Classify features
classification = taxonomy.classify_features(feature_list, timestamps)

# Get clean features only
clean_features = taxonomy.get_clean_features()
```

### 2. Dual-Platform Validation

- **TravisTorrent:** 100,000 builds from 1,283 GitHub projects (2013-2017)
- **GHALogs:** 75,706 workflow runs from 7,620 repositories (2023)
- **11-year span:** Validates temporal robustness across major CI/CD evolution

### 3. Complete Reproducibility

- All random seeds documented (42 throughout)
- All hyperparameters specified
- All software versions listed
- End-to-end reproduction scripts

---

## Documentation

Complete documentation available in `docs/`:

- **[INSTALLATION.md](docs/INSTALLATION.md)** - Detailed setup guide
- **[USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - How to run all experiments
- **[METHODOLOGY.md](docs/METHODOLOGY.md)** - Detailed methodology explanation
- **[LEAKAGE_TAXONOMY.md](docs/LEAKAGE_TAXONOMY.md)** - 3-type taxonomy with examples
- **[FEATURE_MAPPINGS.md](docs/FEATURE_MAPPINGS.md)** - Platform-specific feature mappings
- **[HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md)** - All model configurations
- **[STATISTICAL_METHODS.md](docs/STATISTICAL_METHODS.md)** - Statistical tests explained
- **[FAQ.md](docs/FAQ.md)** - Frequently asked questions

---

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{rangari2025temporal,
  title={A taxonomy for detecting and preventing temporal data leakage in
         machine learning-based build prediction: A dual-platform empirical validation},
  author={Rangari, Amit and Mishra, Lalit Narayan and
          Nagrare, Sandesh and Nayak, Saroj Kumar},
  journal={PLOS ONE},
  volume={XX},
  number={X},
  pages={eXXXXXXX},
  year={2025},
  publisher={Public Library of Science},
  doi={10.1371/journal.pone.XXXXXXX}
}
```

**Dataset Citations:**

```bibtex
@inproceedings{beller2017travistorrent,
  title={TravisTorrent: Synthesizing Travis CI and GitHub for
         Full-Stack Research on Continuous Integration},
  author={Beller, Moritz and Gousios, Georgios and Zaidman, Andy},
  booktitle={Proceedings of the 14th International Conference on
             Mining Software Repositories (MSR)},
  pages={447--450},
  year={2017},
  doi={10.1109/MSR.2017.24}
}

@misc{ghalogs2024,
  title={GHALogs: A Large-Scale Dataset of GitHub Actions Logs},
  author={Decan, Alexandre and Mens, Tom},
  year={2024},
  doi={10.5281/zenodo.10154920}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Corresponding Author:** Lalit Narayan Mishra
- 📧 Email: lnm8910@gmail.com
- 🏢 Affiliation: Lowe's Companies, Inc., Charlotte, NC, USA

**Co-Authors:**
- Amit Rangari (amitrangari@gmail.com) - JPMorgan Chase & Co
- Sandesh Nagrare (sandesh.nagrare@vit.edu.in) - Vidyalankar Institute of Technology
- Saroj Kumar Nayak (saroj.nayak@cb.amrita.edu) - Amrita School of Computing

---

## Acknowledgments

- **TravisTorrent Team:** Moritz Beller, Georgios Gousios, Andy Zaidman
- **GHALogs Team:** Alexandre Decan, Tom Mens
- **PLOS ONE:** For publishing our research
- **Open Source Community:** For the tools and datasets that made this research possible

---

## Repository Stats

![GitHub stars](https://img.shields.io/github/stars/lnm8910/plos-one-build-prediction?style=social)
![GitHub forks](https://img.shields.io/github/forks/lnm8910/plos-one-build-prediction?style=social)
![GitHub issues](https://img.shields.io/github/issues/lnm8910/plos-one-build-prediction)

---

**🔖 Keywords:** Build Prediction, CI/CD, Continuous Integration, Data Leakage, Machine Learning, Temporal Validation, TravisTorrent, GitHub Actions, Software Engineering, DevOps, Reproducibility
