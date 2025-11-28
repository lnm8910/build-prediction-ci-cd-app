"""
Machine Learning Models Module

Implements model training and evaluation for build outcome prediction:
- Random Forest (primary model, 82.73-83.30% accuracy)
- Gradient Boosting (ensemble baseline)
- Logistic Regression (linear baseline)
- Decision Tree (single tree baseline)

Features:
- Hyperparameter tuning with grid search
- Time-series cross-validation
- Class imbalance handling (balanced weights)
- Model persistence and loading
- Reproducible training (fixed random seed)

Main Components:
- TravisTorrentTrainer: Train models on Travis CI data
- GHALogsTrainer: Train models on GitHub Actions data
- ModelEvaluator: Comprehensive evaluation metrics

Authors: Amit Rangari, Lalit Narayan Mishra, Sandesh Nagrare, Saroj Kumar Nayak
"""

__version__ = '1.0.0'
