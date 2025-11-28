"""
Correlation Analyzer for Data Leakage Detection

Detects suspiciously high correlations between features and target variables
that may indicate direct outcome encoding (Type 1 leakage).

High correlation (r > 0.9) between a feature and the target often suggests
the feature directly encodes the outcome being predicted, making it unusable
for prospective prediction.

Example:
    >>> analyzer = CorrelationAnalyzer(threshold=0.9)
    >>> report = analyzer.analyze(data, target='build_success')
    >>> suspicious = analyzer.get_suspicious_features(report)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from scipy import stats
import warnings


@dataclass
class CorrelationReport:
    """Report of correlation analysis results"""
    feature_name: str
    correlation: float
    p_value: float
    is_suspicious: bool
    correlation_type: str  # 'pearson', 'spearman', or 'point-biserial'
    sample_size: int
    confidence_interval: Tuple[float, float] = None
    interpretation: str = ""

    def __str__(self) -> str:
        status = "⚠️ SUSPICIOUS" if self.is_suspicious else "✓ OK"
        return (f"{self.feature_name}: {status} "
                f"(r={self.correlation:.4f}, p={self.p_value:.4e}, "
                f"{self.correlation_type})")


@dataclass
class CorrelationMatrix:
    """Full correlation analysis results"""
    target_column: str
    threshold: float
    total_features: int
    suspicious_count: int
    reports: Dict[str, CorrelationReport] = field(default_factory=dict)
    correlation_matrix: pd.DataFrame = None

    def get_suspicious_features(self) -> List[str]:
        """Get list of features with suspicious correlations"""
        return [name for name, report in self.reports.items()
                if report.is_suspicious]

    def get_safe_features(self) -> List[str]:
        """Get list of features with safe correlations"""
        return [name for name, report in self.reports.items()
                if not report.is_suspicious]


class CorrelationAnalyzer:
    """
    Analyzes correlations between features and target to detect potential leakage.

    Suspiciously high correlations (typically r > 0.9) often indicate that
    a feature directly encodes the outcome being predicted, which violates
    the principle of prospective prediction.

    Attributes:
        threshold: Correlation threshold above which features are flagged (default: 0.9)
        method: Correlation method ('pearson', 'spearman', or 'auto')
        confidence_level: Confidence level for intervals (default: 0.95)
    """

    def __init__(self,
                 threshold: float = 0.9,
                 method: str = 'auto',
                 confidence_level: float = 0.95,
                 verbose: bool = False):
        """
        Initialize correlation analyzer.

        Args:
            threshold: Absolute correlation value above which to flag features
            method: 'pearson', 'spearman', 'point_biserial', or 'auto'
            confidence_level: Confidence level for intervals (0.95 = 95%)
            verbose: If True, print detailed analysis
        """
        self.threshold = threshold
        self.method = method
        self.confidence_level = confidence_level
        self.verbose = verbose

    def analyze(self,
                data: pd.DataFrame,
                target: str,
                features: Optional[List[str]] = None) -> CorrelationMatrix:
        """
        Analyze correlations between features and target.

        Args:
            data: DataFrame containing features and target
            target: Name of target column
            features: List of feature names to analyze (if None, use all except target)

        Returns:
            CorrelationMatrix with detailed results
        """
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data")

        if features is None:
            features = [col for col in data.columns if col != target]

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"CORRELATION ANALYSIS: {len(features)} features vs {target}")
            print(f"Threshold: |r| > {self.threshold} (suspicious)")
            print(f"{'='*80}\n")

        reports = {}
        suspicious_count = 0

        for feature in features:
            report = self._analyze_single_feature(data, feature, target)
            reports[feature] = report
            if report.is_suspicious:
                suspicious_count += 1

            if self.verbose and report.is_suspicious:
                print(f"⚠️  {report}")

        # Create correlation matrix for all features
        corr_matrix = self._create_correlation_matrix(data, features + [target])

        result = CorrelationMatrix(
            target_column=target,
            threshold=self.threshold,
            total_features=len(features),
            suspicious_count=suspicious_count,
            reports=reports,
            correlation_matrix=corr_matrix
        )

        if self.verbose:
            self._print_summary(result)

        return result

    def _analyze_single_feature(self,
                                data: pd.DataFrame,
                                feature: str,
                                target: str) -> CorrelationReport:
        """Analyze correlation for a single feature"""
        # Determine appropriate correlation method
        method = self._select_correlation_method(data[feature], data[target])

        # Compute correlation and p-value
        if method == 'pearson':
            corr, p_value = stats.pearsonr(
                self._clean_numeric(data[feature]),
                self._clean_numeric(data[target])
            )
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(
                self._clean_numeric(data[feature]),
                self._clean_numeric(data[target])
            )
        elif method == 'point_biserial':
            # For binary targets
            corr, p_value = stats.pointbiserialr(
                self._clean_numeric(data[target]),
                self._clean_numeric(data[feature])
            )
        else:
            # Fallback to converting to numeric and using Pearson
            corr, p_value = stats.pearsonr(
                pd.factorize(data[feature])[0],
                pd.factorize(data[target])[0]
            )

        # Compute confidence interval
        ci = self._compute_confidence_interval(
            corr,
            len(data),
            self.confidence_level
        )

        # Determine if suspicious
        is_suspicious = abs(corr) >= self.threshold

        # Generate interpretation
        interpretation = self._interpret_correlation(corr, p_value, is_suspicious)

        return CorrelationReport(
            feature_name=feature,
            correlation=corr,
            p_value=p_value,
            is_suspicious=is_suspicious,
            correlation_type=method,
            sample_size=len(data),
            confidence_interval=ci,
            interpretation=interpretation
        )

    def _select_correlation_method(self,
                                   feature: pd.Series,
                                   target: pd.Series) -> str:
        """Select appropriate correlation method based on data types"""
        if self.method != 'auto':
            return self.method

        # Check if binary target (classification task)
        target_unique = target.nunique()
        feature_unique = feature.nunique()

        if target_unique == 2 and self._is_numeric(feature):
            return 'point_biserial'
        elif self._is_numeric(feature) and self._is_numeric(target):
            # Check for normality (if non-normal, use Spearman)
            if len(feature) < 5000:  # Only check normality for smaller datasets
                _, p_feat = stats.shapiro(feature.dropna().sample(min(5000, len(feature.dropna()))))
                _, p_targ = stats.shapiro(target.dropna().sample(min(5000, len(target.dropna()))))
                if p_feat < 0.05 or p_targ < 0.05:
                    return 'spearman'
            return 'pearson'
        else:
            return 'spearman'  # Robust to non-normal data

    def _is_numeric(self, series: pd.Series) -> bool:
        """Check if series is numeric"""
        return pd.api.types.is_numeric_dtype(series)

    def _clean_numeric(self, series: pd.Series) -> np.ndarray:
        """Convert series to clean numeric array"""
        if self._is_numeric(series):
            return series.fillna(series.median()).values
        else:
            # Convert categorical to numeric codes
            return pd.factorize(series)[0]

    def _compute_confidence_interval(self,
                                     r: float,
                                     n: int,
                                     confidence: float) -> Tuple[float, float]:
        """
        Compute confidence interval for correlation coefficient using Fisher's Z transform.

        Args:
            r: Correlation coefficient
            n: Sample size
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if abs(r) >= 1.0:
            return (r, r)  # Perfect correlation, no interval

        # Fisher's Z transformation
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1 / np.sqrt(n - 3)

        # Z-score for confidence level
        z_crit = stats.norm.ppf((1 + confidence) / 2)

        # Confidence interval in Z space
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se

        # Transform back to r space
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

        return (r_lower, r_upper)

    def _interpret_correlation(self,
                               r: float,
                               p_value: float,
                               is_suspicious: bool) -> str:
        """Generate human-readable interpretation"""
        abs_r = abs(r)

        # Strength interpretation (Cohen's guidelines)
        if abs_r >= 0.9:
            strength = "very strong"
        elif abs_r >= 0.7:
            strength = "strong"
        elif abs_r >= 0.5:
            strength = "moderate"
        elif abs_r >= 0.3:
            strength = "weak"
        else:
            strength = "very weak"

        direction = "positive" if r > 0 else "negative"
        significance = "significant" if p_value < 0.05 else "not significant"

        interpretation = f"{strength.capitalize()} {direction} correlation ({significance})"

        if is_suspicious:
            interpretation += " - LIKELY LEAKAGE: This correlation is suspiciously high and suggests direct outcome encoding"

        return interpretation

    def _create_correlation_matrix(self,
                                   data: pd.DataFrame,
                                   columns: List[str]) -> pd.DataFrame:
        """Create full correlation matrix"""
        # Select only numeric columns or convert
        numeric_data = pd.DataFrame()
        for col in columns:
            if self._is_numeric(data[col]):
                numeric_data[col] = data[col].fillna(data[col].median())
            else:
                numeric_data[col] = pd.factorize(data[col])[0]

        return numeric_data.corr()

    def _print_summary(self, result: CorrelationMatrix):
        """Print summary of correlation analysis"""
        print(f"\n{'='*80}")
        print("CORRELATION ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"Target: {result.target_column}")
        print(f"Total features analyzed: {result.total_features}")
        print(f"Suspicious features (|r| > {result.threshold}): {result.suspicious_count} ({result.suspicious_count/result.total_features*100:.1f}%)")

        if result.suspicious_count > 0:
            print(f"\nSUSPICIOUS FEATURES:")
            for name in sorted(result.get_suspicious_features()):
                report = result.reports[name]
                print(f"  ⚠️  {name:<40} r={report.correlation:>7.4f} (p={report.p_value:.2e})")

        print(f"{'='*80}\n")

    def get_suspicious_features(self, result: CorrelationMatrix) -> List[str]:
        """Extract list of suspicious feature names"""
        return result.get_suspicious_features()

    def get_safe_features(self, result: CorrelationMatrix) -> List[str]:
        """Extract list of safe feature names"""
        return result.get_safe_features()

    def export_report(self,
                     result: CorrelationMatrix,
                     output_path: str,
                     format: str = 'csv'):
        """
        Export correlation analysis to file.

        Args:
            result: CorrelationMatrix from analyze()
            output_path: Path to save results
            format: 'csv', 'json', or 'markdown'
        """
        data = []
        for name, report in result.reports.items():
            ci_lower, ci_upper = report.confidence_interval if report.confidence_interval else (None, None)
            data.append({
                'feature': name,
                'correlation': report.correlation,
                'abs_correlation': abs(report.correlation),
                'p_value': report.p_value,
                'is_suspicious': report.is_suspicious,
                'correlation_type': report.correlation_type,
                'sample_size': report.sample_size,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'interpretation': report.interpretation
            })

        df = pd.DataFrame(data).sort_values('abs_correlation', ascending=False)

        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'markdown':
            df.to_markdown(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"✓ Correlation analysis exported to: {output_path}")

    def plot_correlations(self,
                         result: CorrelationMatrix,
                         output_path: Optional[str] = None,
                         top_n: int = 20):
        """
        Plot correlation analysis results.

        Args:
            result: CorrelationMatrix from analyze()
            output_path: Path to save plot (if None, display only)
            top_n: Number of top correlations to plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            warnings.warn("matplotlib and seaborn required for plotting. Skipping plot.")
            return

        # Get top N features by absolute correlation
        sorted_reports = sorted(
            result.reports.items(),
            key=lambda x: abs(x[1].correlation),
            reverse=True
        )[:top_n]

        features = [name for name, _ in sorted_reports]
        correlations = [report.correlation for _, report in sorted_reports]
        is_suspicious = [report.is_suspicious for _, report in sorted_reports]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))

        colors = ['red' if susp else 'blue' for susp in is_suspicious]
        bars = ax.barh(features, correlations, color=colors, alpha=0.6)

        # Add threshold lines
        ax.axvline(x=result.threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold (±{result.threshold})')
        ax.axvline(x=-result.threshold, color='red', linestyle='--', alpha=0.5)

        ax.set_xlabel('Correlation with Target', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Correlations with {result.target_column}', fontsize=14, fontweight='bold')
        ax.set_xlim(-1.1, 1.1)
        ax.grid(axis='x', alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Correlation plot saved to: {output_path}")
        else:
            plt.show()

        plt.close()
