"""
Model Evaluation Module

Comprehensive evaluation framework with statistical rigor:
- Performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Statistical significance testing (Wilcoxon, Kruskal-Wallis)
- Effect size computation (Cohen's d, Cohen's h, η²)
- Confidence intervals (Bootstrap resampling)
- Multiple testing corrections (Bonferroni)

Main Components:
- MetricsCalculator: Compute all evaluation metrics
- StatisticalTests: Hypothesis testing and effect sizes
- ConfidenceIntervals: Bootstrap and parametric CIs
- CrossValidator: Time-series cross-validation

Authors: Amit Rangari, Lalit Narayan Mishra, Sandesh Nagrare, Saroj Kumar Nayak
"""

__version__ = '1.0.0'
