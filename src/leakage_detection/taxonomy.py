"""
Temporal Data Leakage Taxonomy - 3-Type Classification System

Implements the systematic 3-type taxonomy for detecting temporal data leakage
in machine learning-based software engineering prediction tasks.

Type 1: Direct Outcome Encoding
    - Features that explicitly encode the outcome being predicted
    - Examples: tr_status (build status), test_failed_count (for test prediction)
    - Detection: Feature name contains outcome-related keywords or perfect correlation

Type 2: Execution-Dependent Metrics
    - Features that can only be computed after the predicted event occurs
    - Examples: build_duration, test_execution_time, code_coverage
    - Detection: Requires execution completion to compute value

Type 3: Future Information Leakage
    - Features that incorporate information from after the prediction timestamp
    - Examples: repo_stars (changes after build), rolling_averages (using future data)
    - Detection: Timestamp analysis reveals future data dependency

Reference:
    Rangari, A., Mishra, L.N., Nagrare, S., Nayak, S.K. (2025).
    "A taxonomy for detecting and preventing temporal data leakage in
    machine learning-based build prediction"
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
import re
import warnings


class LeakageType(Enum):
    """Enumeration of temporal data leakage types"""
    CLEAN = "CLEAN"                      # No leakage detected
    TYPE_1_DIRECT_OUTCOME = "TYPE_1"     # Direct outcome encoding
    TYPE_2_EXECUTION_DEPENDENT = "TYPE_2"  # Execution-dependent metrics
    TYPE_3_FUTURE_INFORMATION = "TYPE_3"  # Future information leakage
    UNKNOWN = "UNKNOWN"                   # Cannot be classified


@dataclass
class FeatureClassification:
    """Classification result for a single feature"""
    feature_name: str
    leakage_type: LeakageType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    detected_by: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (f"{self.feature_name}: {self.leakage_type.value} "
                f"(confidence={self.confidence:.2f}, {self.reasoning})")


class TemporalLeakageTaxonomy:
    """
    Main class for detecting and classifying temporal data leakage.

    This implements the 3-type taxonomy described in our PLOS ONE paper,
    providing automated detection of temporal leakage patterns.

    Attributes:
        strict_mode (bool): If True, flags ambiguous cases as leaky
        correlation_threshold (float): Correlation threshold for Type 1 detection (default: 0.9)
        verbose (bool): If True, prints detailed classification reasoning

    Example:
        >>> taxonomy = TemporalLeakageTaxonomy()
        >>> results = taxonomy.classify_features(df, target_column='build_success')
        >>> clean_features = taxonomy.get_clean_features(results)
        >>> leaky_features = taxonomy.get_leaky_features(results)
    """

    # Type 1: Keywords indicating direct outcome encoding
    TYPE_1_KEYWORDS = {
        'status', 'result', 'outcome', 'success', 'failure', 'failed', 'passed',
        'error', 'exception', 'crash', 'abort', 'cancel', 'skip', 'verdict'
    }

    # Type 2: Keywords indicating execution-dependent metrics
    TYPE_2_KEYWORDS = {
        'duration', 'time', 'runtime', 'execution', 'elapsed', 'latency',
        'coverage', 'tested', 'ran', 'executed', 'invoked', 'performance',
        'memory', 'cpu', 'resource', 'log', 'output', 'console'
    }

    # Type 3: Keywords indicating future/time-dependent information
    TYPE_3_KEYWORDS = {
        'stars', 'forks', 'watchers', 'subscribers', 'popularity',
        'downloads', 'views', 'followers', 'trending', 'recent'
    }

    # Safe keywords that are pre-build accessible
    SAFE_KEYWORDS = {
        'previous', 'prior', 'last', 'historical', 'past', 'before',
        'project', 'repository', 'repo', 'code', 'commit', 'author',
        'file', 'line', 'change', 'added', 'deleted', 'modified',
        'count', 'number', 'total', 'avg', 'mean', 'median', 'max', 'min'
    }

    def __init__(self,
                 strict_mode: bool = False,
                 correlation_threshold: float = 0.9,
                 verbose: bool = False):
        """
        Initialize the taxonomy classifier.

        Args:
            strict_mode: If True, classify ambiguous features as leaky
            correlation_threshold: Threshold for detecting Type 1 via correlation
            verbose: If True, print detailed reasoning
        """
        self.strict_mode = strict_mode
        self.correlation_threshold = correlation_threshold
        self.verbose = verbose
        self._classification_cache: Dict[str, FeatureClassification] = {}

    def classify_features(self,
                         data: pd.DataFrame,
                         target_column: Optional[str] = None,
                         timestamp_column: Optional[str] = None,
                         feature_metadata: Optional[Dict[str, Dict]] = None) -> Dict[str, FeatureClassification]:
        """
        Classify all features in a dataset according to the 3-type taxonomy.

        Args:
            data: DataFrame containing features and optionally target
            target_column: Name of target variable (for correlation analysis)
            timestamp_column: Name of timestamp column (for temporal analysis)
            feature_metadata: Optional dict with additional feature information
                Format: {feature_name: {'description': str, 'source': str, ...}}

        Returns:
            Dictionary mapping feature names to FeatureClassification objects
        """
        if feature_metadata is None:
            feature_metadata = {}

        results = {}
        features_to_classify = [col for col in data.columns
                               if col not in [target_column, timestamp_column]]

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"TEMPORAL LEAKAGE TAXONOMY - CLASSIFYING {len(features_to_classify)} FEATURES")
            print(f"{'='*80}\n")

        for feature in features_to_classify:
            # Check cache first
            cache_key = f"{feature}_{target_column}_{timestamp_column}"
            if cache_key in self._classification_cache:
                results[feature] = self._classification_cache[cache_key]
                continue

            # Perform classification
            classification = self._classify_single_feature(
                feature,
                data,
                target_column,
                timestamp_column,
                feature_metadata.get(feature, {})
            )

            results[feature] = classification
            self._classification_cache[cache_key] = classification

            if self.verbose:
                print(classification)

        # Summary statistics
        if self.verbose:
            self._print_classification_summary(results)

        return results

    def _classify_single_feature(self,
                                feature_name: str,
                                data: pd.DataFrame,
                                target_column: Optional[str],
                                timestamp_column: Optional[str],
                                metadata: Dict) -> FeatureClassification:
        """
        Classify a single feature using multiple detection strategies.

        Detection pipeline:
        1. Name-based keyword detection (Type 1, 2, 3)
        2. Correlation analysis with target (Type 1)
        3. Temporal dependency analysis (Type 3)
        4. Metadata-based classification
        5. Default to CLEAN if no leakage detected
        """
        detected_by = []
        leakage_type = LeakageType.CLEAN
        confidence = 0.0
        reasoning = "No leakage detected"

        feature_lower = feature_name.lower()

        # Strategy 1: Name-based keyword detection for Type 1
        type1_score = self._check_keywords(feature_lower, self.TYPE_1_KEYWORDS)
        if type1_score > 0:
            leakage_type = LeakageType.TYPE_1_DIRECT_OUTCOME
            confidence = min(1.0, type1_score / 2.0)  # Scale keyword count
            reasoning = f"Feature name contains outcome keywords: {self._get_matched_keywords(feature_lower, self.TYPE_1_KEYWORDS)}"
            detected_by.append("keyword_type1")

        # Strategy 2: Name-based keyword detection for Type 2
        if leakage_type == LeakageType.CLEAN:
            type2_score = self._check_keywords(feature_lower, self.TYPE_2_KEYWORDS)
            if type2_score > 0:
                leakage_type = LeakageType.TYPE_2_EXECUTION_DEPENDENT
                confidence = min(1.0, type2_score / 2.0)
                reasoning = f"Feature name contains execution-dependent keywords: {self._get_matched_keywords(feature_lower, self.TYPE_2_KEYWORDS)}"
                detected_by.append("keyword_type2")

        # Strategy 3: Name-based keyword detection for Type 3
        if leakage_type == LeakageType.CLEAN:
            type3_score = self._check_keywords(feature_lower, self.TYPE_3_KEYWORDS)
            if type3_score > 0:
                leakage_type = LeakageType.TYPE_3_FUTURE_INFORMATION
                confidence = min(1.0, type3_score / 2.0)
                reasoning = f"Feature name contains future/time-dependent keywords: {self._get_matched_keywords(feature_lower, self.TYPE_3_KEYWORDS)}"
                detected_by.append("keyword_type3")

        # Strategy 4: Correlation analysis for Type 1 (if target provided)
        if target_column and target_column in data.columns:
            corr = self._compute_correlation(data[feature_name], data[target_column])
            if abs(corr) >= self.correlation_threshold:
                if leakage_type == LeakageType.CLEAN or confidence < 0.9:
                    leakage_type = LeakageType.TYPE_1_DIRECT_OUTCOME
                    confidence = max(confidence, abs(corr))
                    reasoning = f"High correlation with target ({corr:.4f} >= {self.correlation_threshold})"
                    detected_by.append("correlation")

        # Strategy 5: Check for safe keywords (reduces false positives)
        safe_score = self._check_keywords(feature_lower, self.SAFE_KEYWORDS)
        if safe_score > 0 and leakage_type == LeakageType.CLEAN:
            confidence = min(1.0, safe_score / 3.0)
            reasoning = f"Pre-build accessible: {self._get_matched_keywords(feature_lower, self.SAFE_KEYWORDS)}"
            detected_by.append("safe_keywords")

        # Strategy 6: Metadata-based classification
        if metadata:
            meta_classification = self._classify_from_metadata(metadata)
            if meta_classification[0] != LeakageType.CLEAN:
                leakage_type = meta_classification[0]
                confidence = max(confidence, meta_classification[1])
                reasoning = f"Metadata indicates: {meta_classification[2]}"
                detected_by.append("metadata")

        # Strategy 7: Strict mode - flag ambiguous features
        if self.strict_mode and confidence > 0 and confidence < 0.5:
            leakage_type = LeakageType.UNKNOWN
            reasoning = f"Ambiguous classification (strict mode): {reasoning}"

        # Default to CLEAN with low confidence if nothing detected
        if leakage_type == LeakageType.CLEAN and not detected_by:
            confidence = 0.3  # Low confidence for default classification

        return FeatureClassification(
            feature_name=feature_name,
            leakage_type=leakage_type,
            confidence=confidence,
            reasoning=reasoning,
            detected_by=detected_by,
            metadata=metadata
        )

    def _check_keywords(self, text: str, keywords: Set[str]) -> int:
        """Count how many keywords from set appear in text"""
        return sum(1 for keyword in keywords if keyword in text)

    def _get_matched_keywords(self, text: str, keywords: Set[str]) -> List[str]:
        """Return list of keywords that appear in text"""
        return [keyword for keyword in keywords if keyword in text]

    def _compute_correlation(self, feature: pd.Series, target: pd.Series) -> float:
        """Compute correlation between feature and target, handling categorical variables"""
        try:
            # If categorical, convert to numeric
            if feature.dtype == 'object' or target.dtype == 'object':
                feature_num = pd.factorize(feature)[0]
                target_num = pd.factorize(target)[0] if target.dtype == 'object' else target
                return np.corrcoef(feature_num, target_num)[0, 1]
            else:
                return feature.corr(target)
        except Exception:
            return 0.0

    def _classify_from_metadata(self, metadata: Dict) -> Tuple[LeakageType, float, str]:
        """Classify based on provided metadata"""
        if 'leakage_type' in metadata:
            type_str = metadata['leakage_type'].upper()
            if 'TYPE_1' in type_str or 'DIRECT' in type_str:
                return LeakageType.TYPE_1_DIRECT_OUTCOME, 1.0, "Explicitly marked as Type 1"
            elif 'TYPE_2' in type_str or 'EXECUTION' in type_str:
                return LeakageType.TYPE_2_EXECUTION_DEPENDENT, 1.0, "Explicitly marked as Type 2"
            elif 'TYPE_3' in type_str or 'FUTURE' in type_str:
                return LeakageType.TYPE_3_FUTURE_INFORMATION, 1.0, "Explicitly marked as Type 3"

        if 'availability' in metadata:
            avail = metadata['availability'].lower()
            if 'post-build' in avail or 'execution' in avail:
                return LeakageType.TYPE_2_EXECUTION_DEPENDENT, 0.9, f"Availability: {avail}"
            elif 'future' in avail or 'time-dependent' in avail:
                return LeakageType.TYPE_3_FUTURE_INFORMATION, 0.9, f"Availability: {avail}"

        return LeakageType.CLEAN, 0.0, ""

    def _print_classification_summary(self, results: Dict[str, FeatureClassification]):
        """Print summary statistics of classification"""
        type_counts = {leakage_type: 0 for leakage_type in LeakageType}
        for classification in results.values():
            type_counts[classification.leakage_type] += 1

        total = len(results)
        print(f"\n{'='*80}")
        print("CLASSIFICATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total features classified: {total}")
        print(f"\nCLEAN features:             {type_counts[LeakageType.CLEAN]:>4} ({type_counts[LeakageType.CLEAN]/total*100:>5.1f}%)")
        print(f"TYPE 1 (Direct Outcome):    {type_counts[LeakageType.TYPE_1_DIRECT_OUTCOME]:>4} ({type_counts[LeakageType.TYPE_1_DIRECT_OUTCOME]/total*100:>5.1f}%)")
        print(f"TYPE 2 (Execution-Dep):     {type_counts[LeakageType.TYPE_2_EXECUTION_DEPENDENT]:>4} ({type_counts[LeakageType.TYPE_2_EXECUTION_DEPENDENT]/total*100:>5.1f}%)")
        print(f"TYPE 3 (Future Info):       {type_counts[LeakageType.TYPE_3_FUTURE_INFORMATION]:>4} ({type_counts[LeakageType.TYPE_3_FUTURE_INFORMATION]/total*100:>5.1f}%)")
        print(f"UNKNOWN:                    {type_counts[LeakageType.UNKNOWN]:>4} ({type_counts[LeakageType.UNKNOWN]/total*100:>5.1f}%)")
        print(f"{'='*80}\n")

    def get_clean_features(self, classifications: Dict[str, FeatureClassification]) -> List[str]:
        """
        Extract list of clean (non-leaky) feature names.

        Args:
            classifications: Results from classify_features()

        Returns:
            List of feature names classified as CLEAN
        """
        return [name for name, cls in classifications.items()
                if cls.leakage_type == LeakageType.CLEAN]

    def get_leaky_features(self,
                          classifications: Dict[str, FeatureClassification],
                          include_types: Optional[List[LeakageType]] = None) -> Dict[str, FeatureClassification]:
        """
        Extract leaky features, optionally filtered by type.

        Args:
            classifications: Results from classify_features()
            include_types: If provided, only include these leakage types

        Returns:
            Dictionary of leaky feature classifications
        """
        if include_types is None:
            include_types = [
                LeakageType.TYPE_1_DIRECT_OUTCOME,
                LeakageType.TYPE_2_EXECUTION_DEPENDENT,
                LeakageType.TYPE_3_FUTURE_INFORMATION
            ]

        return {name: cls for name, cls in classifications.items()
                if cls.leakage_type in include_types}

    def export_classification(self,
                             classifications: Dict[str, FeatureClassification],
                             output_path: str,
                             format: str = 'csv'):
        """
        Export classification results to file.

        Args:
            classifications: Results from classify_features()
            output_path: Path to save results
            format: 'csv', 'json', or 'markdown'
        """
        # Convert to DataFrame
        data = []
        for name, cls in classifications.items():
            data.append({
                'feature': name,
                'leakage_type': cls.leakage_type.value,
                'confidence': cls.confidence,
                'reasoning': cls.reasoning,
                'detected_by': ','.join(cls.detected_by)
            })

        df = pd.DataFrame(data)

        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'markdown':
            df.to_markdown(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"✓ Classification results exported to: {output_path}")

    def clear_cache(self):
        """Clear the classification cache"""
        self._classification_cache.clear()
