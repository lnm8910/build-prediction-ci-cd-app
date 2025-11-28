"""
Build Prediction Source Code Package

This package contains all source code for the PLOS ONE paper:
"A taxonomy for detecting and preventing temporal data leakage in
machine learning-based build prediction: A dual-platform empirical validation"

Modules:
- leakage_detection: Core 3-type temporal leakage taxonomy toolkit
- preprocessing: Data preprocessing for TravisTorrent and GHALogs
- models: Machine learning model training (Random Forest, Gradient Boosting, Logistic Regression)
- evaluation: Model evaluation with statistical tests
- visualization: Figure generation for publication

Authors: Amit Rangari, Lalit Narayan Mishra, Sandesh Nagrare, Saroj Kumar Nayak
License: MIT
"""

__version__ = '1.0.0'
__author__ = 'Amit Rangari, Lalit Narayan Mishra, Sandesh Nagrare, Saroj Kumar Nayak'

# Import main classes for convenient access
from .leakage_detection import TemporalLeakageTaxonomy, FeatureValidator, CorrelationAnalyzer

__all__ = [
    'TemporalLeakageTaxonomy',
    'FeatureValidator',
    'CorrelationAnalyzer'
]
