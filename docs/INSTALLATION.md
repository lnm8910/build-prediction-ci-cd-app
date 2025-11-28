# Installation Guide

## System Requirements

**Operating System:**
- Linux (recommended - Ubuntu 20.04+, Debian 11+, Fedora 35+)
- macOS (10.15+)
- Windows (10/11 with WSL2 recommended)

**Hardware (Minimum):**
- CPU: 2 cores
- RAM: 8 GB
- Disk: 5 GB free space

**Hardware (Recommended):**
- CPU: 4+ cores (for parallel processing)
- RAM: 16 GB
- Disk: 10 GB free space

**Software:**
- Python 3.8 or higher
- pip 20.0+ or conda 4.9+

---

## Installation Methods

### Method 1: Using pip (Recommended)

```bash
# 1. Download/clone this repository
# If from Zenodo: extract the archive
unzip build-prediction-ci-cd-app.zip
cd build-prediction-ci-cd-app

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "from src.leakage_detection import TemporalLeakageTaxonomy; print('✓ Installation successful!')"
```

### Method 2: Using conda

```bash
# 1. Create conda environment
conda create -n build-prediction python=3.9 -y
conda activate build-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__} installed')"
```

---

## Verification Steps

### Test 1: Import All Modules

```python
# Should run without errors
from src.leakage_detection import TemporalLeakageTaxonomy, FeatureValidator, CorrelationAnalyzer
from src.models import train_travistorrent, train_ghalogs
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("✓ All imports successful!")
```

### Test 2: Load Sample Dataset

```python
# Load TravisTorrent
df = pd.read_csv('data/travistorrent/travistorrent_100k_all_features.csv')
print(f"✓ Loaded {len(df)} builds with {len(df.columns)} features")

# Expected: 100,000 builds, 77 features
```

### Test 3: Run Leakage Toolkit

```python
from src.leakage_detection import TemporalLeakageTaxonomy

taxonomy = TemporalLeakageTaxonomy(verbose=True)
classifications = taxonomy.classify_features(
    data=df.head(1000),  # Test on small sample
    target_column='build_success'
)

clean = taxonomy.get_clean_features(classifications)
print(f"✓ Taxonomy working! Found {len(clean)} clean features")

# Expected: ~31 clean features
```

---

## Troubleshooting

### Issue: ModuleNotFoundError

```bash
# Solution: Add repository root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

### Issue: Permission denied

```bash
# Solution: Make scripts executable
chmod +x experiments/**/*.py
chmod +x experiments/**/*.sh
```

### Issue: scikit-learn version incompatible

```bash
# Solution: Install specific version
pip install scikit-learn==1.3.2
```

---

**Installation time:** 5-10 minutes
**Status:** Complete
