"""
Temporal Data Leakage Detection Toolkit

This module implements a systematic 3-type taxonomy for detecting and preventing
temporal data leakage in machine learning-based software engineering prediction tasks.

Main Components:
- TemporalLeakageTaxonomy: 3-type classifier (Direct Outcome, Execution-Dependent, Future Info)
- FeatureValidator: Temporal availability validation
- CorrelationAnalyzer: High-correlation detection (r > 0.9)

Usage:
    from src.leakage_detection import TemporalLeakageTaxonomy

    taxonomy = TemporalLeakageTaxonomy()
    classification = taxonomy.classify_features(features, metadata)
    clean_features = taxonomy.get_clean_features()

Authors: Amit Rangari, Lalit Narayan Mishra, Sandesh Nagrare, Saroj Kumar Nayak
License: MIT
"""

from .taxonomy import TemporalLeakageTaxonomy, LeakageType
from .feature_validator import FeatureValidator, TemporalAvailability
from .correlation_analyzer import CorrelationAnalyzer, CorrelationReport

__all__ = [
    'TemporalLeakageTaxonomy',
    'LeakageType',
    'FeatureValidator',
    'TemporalAvailability',
    'CorrelationAnalyzer',
    'CorrelationReport'
]

__version__ = '1.0.0'
__author__ = 'Amit Rangari, Lalit Narayan Mishra, Sandesh Nagrare, Saroj Kumar Nayak'
