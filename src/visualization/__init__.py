"""
Visualization Module

Generates all publication-quality figures for the PLOS ONE paper:
- Figure 1: ROC curves comparison (3 models)
- Figure 2: Feature importance ranking (top 10 features)
- Figure 3: Cross-language performance (4 languages)
- Figure 4: Data leakage impact visualization
- Figure 5: Cross-platform comparison (Travis CI vs GitHub Actions)

Output formats:
- PDF (publication quality, 300 DPI)
- PNG (presentations, 300 DPI)

Styling:
- Consistent color schemes
- Publication-ready fonts and sizing
- Clear legends and annotations
- Accessible color palettes (colorblind-friendly)

Main Components:
- FigureGenerator: Base class for all figures
- ROCCurvePlotter: Figure 1 generator
- FeatureImportancePlotter: Figure 2 generator
- CrossLanguagePlotter: Figure 3 generator
- LeakageImpactPlotter: Figure 4 generator
- CrossPlatformPlotter: Figure 5 generator

Authors: Amit Rangari, Lalit Narayan Mishra, Sandesh Nagrare, Saroj Kumar Nayak
"""

__version__ = '1.0.0'
