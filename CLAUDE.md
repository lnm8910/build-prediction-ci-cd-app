# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research replication package for PLOS ONE paper on temporal data leakage prevention in CI/CD build prediction. Analyzes 175,706 builds across TravisTorrent (2013-2017) and GHALogs (2023) datasets using a 3-type temporal leakage taxonomy.

## Common Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Verify installation (runs 6 test checks)
python test_installation.py

# Run cross-platform analysis (main experiment)
python experiments/combined/platform_comparison.py

# Train models
python src/models/train_ghalogs.py --model rf    # rf, gb, lr, dt, or all
python src/models/train_travistorrent.py

# Generate figures
python src/visualization/figure1_roc_curves.py
python src/visualization/figure2_feature_importance.py
python src/visualization/figure3_cross_language.py
python src/visualization/figure4_leakage_impact.py
```

## Architecture

### Core Leakage Detection Toolkit (`src/leakage_detection/`)
The 3-type temporal leakage taxonomy implementation:
- `TemporalLeakageTaxonomy` (`taxonomy.py`): Main classifier using keyword detection + correlation analysis
- `FeatureValidator` (`feature_validator.py`): Validates temporal availability of features
- `CorrelationAnalyzer` (`correlation_analyzer.py`): Detects high-correlation leakage patterns (|r| > 0.9)

**Leakage Types:**
- Type 1 (Direct Outcome): Features explicitly encoding build results (keywords: status, result, failed)
- Type 2 (Execution-Dependent): Features computable only after build execution (keywords: duration, time, coverage)
- Type 3 (Future Information): Features incorporating data from after prediction time (keywords: stars, forks, watchers)

Usage pattern:
```python
from src.leakage_detection import TemporalLeakageTaxonomy, FeatureValidator, CorrelationAnalyzer

taxonomy = TemporalLeakageTaxonomy(strict_mode=False, correlation_threshold=0.9)
classifications = taxonomy.classify_features(data, target_column='conclusion')
clean_features = taxonomy.get_clean_features(classifications)
leaky_features = taxonomy.get_leaky_features(classifications)
taxonomy.export_classification(classifications, 'output.csv')
```

### Data Processing Flow
1. Raw datasets in `data/travistorrent/` and `data/ghalogs/`
2. Preprocessing modules in `src/preprocessing/`
3. Feature classification via leakage detection toolkit
4. Model training with clean vs leaky feature sets
5. Results exported to `results/`

### Model Training Pipeline
Both training scripts (`train_ghalogs.py`, `train_travistorrent.py`) follow the same pattern:
1. Load clean dataset (leakage-free) and leaky dataset (with time-dependent features)
2. Exclude metadata columns: `repo_name`, `head_sha`, `conclusion`, `created_at`, `commit_author`, `repo_default_branch`
3. One-hot encode categorical features with `pd.get_dummies(drop_first=True)`
4. Median imputation for missing values
5. 80/20 train/test split with stratification (`random_state=42`)
6. StandardScaler for feature normalization (fit on train only)
7. Train RandomForest, GradientBoosting, LogisticRegression classifiers
8. Calculate "leakage tax" (accuracy inflation from time-dependent features)

### Default RandomForest Configuration
```python
RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=10,
    min_samples_leaf=4, random_state=42, class_weight='balanced', n_jobs=-1
)
```

## Key Constants

- Random seed: 42 (used throughout for reproducibility)
- Train/test split: 80/20 with stratification
- Correlation threshold for Type 1 detection: 0.9
- Target variable: `conclusion` (GHALogs) or `build_success` (TravisTorrent)
- Expected accuracy: ~82-83% clean, ~97-98% leaky (TravisTorrent)

## Data Files

**Datasets:**
- `data/ghalogs/ghalogs_75k_clean.csv` - 75,706 workflows, 29 clean features
- `data/travistorrent/` - 100K builds, 31 clean features

**Data Dictionaries:**
- `data/ghalogs/ghalogs_data_dictionary.csv`
- `data/travistorrent/travistorrent_data_dictionary.csv`

## Research Questions Mapping

- RQ1: Leakage-free prediction performance → `experiments/*/RQ1_*.py`
- RQ2: Feature importance analysis → `experiments/*/RQ2_*.py`
- RQ3: Cross-language validation → `experiments/travistorrent/RQ3_*.py`
- RQ4: Leakage impact quantification → `experiments/*/RQ4_*.py`
- RQ5: Cross-platform validation → `experiments/combined/platform_comparison.py`
