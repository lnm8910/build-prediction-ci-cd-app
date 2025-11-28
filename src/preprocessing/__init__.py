"""
Data Preprocessing Module

Handles data preprocessing for both TravisTorrent and GHALogs platforms:
- Feature engineering and SDLC mapping
- Missing value imputation
- Feature scaling and normalization
- Temporal train/test splitting
- Data enrichment (GHALogs GitHub API)

Main Components:
- TravisTorrentProcessor: Preprocessing pipeline for Travis CI data
- GHALogsProcessor: Preprocessing pipeline for GitHub Actions data
- FeatureEngineer: SDLC feature mapping and engineering
- DataValidator: Data quality validation

Authors: Amit Rangari, Lalit Narayan Mishra, Sandesh Nagrare, Saroj Kumar Nayak
"""

__version__ = '1.0.0'
