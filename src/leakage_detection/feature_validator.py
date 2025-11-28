"""
Temporal Feature Availability Validator

Validates whether features can be computed at prediction time by analyzing
temporal dependencies, data flow, and computation requirements.

This module complements the TemporalLeakageTaxonomy by providing detailed
temporal availability analysis for individual features.

Key Concepts:
- Prediction Time: The point in time when we need to make a prediction
- Feature Availability: Whether feature value can be computed at prediction time
- Temporal Dependency: Features that depend on future data or execution results

Example:
    >>> validator = FeatureValidator()
    >>> availability = validator.validate_feature('build_duration', 'before_build')
    >>> print(availability.is_available)  # False - requires execution
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta


class TemporalAvailability(Enum):
    """Feature availability at different temporal points"""
    BEFORE_BUILD = "BEFORE_BUILD"           # Available before build starts
    DURING_BUILD = "DURING_BUILD"           # Available while build is running
    AFTER_BUILD = "AFTER_BUILD"             # Available only after build completes
    TIME_DEPENDENT = "TIME_DEPENDENT"       # Changes over time (stars, forks)
    ALWAYS_AVAILABLE = "ALWAYS_AVAILABLE"   # Static metadata (repo name, language)
    UNKNOWN = "UNKNOWN"                     # Cannot determine


@dataclass
class AvailabilityReport:
    """Detailed report on feature temporal availability"""
    feature_name: str
    availability: TemporalAvailability
    is_available_at_prediction: bool
    reasoning: str
    dependencies: List[str]
    computation_time: Optional[str] = None
    earliest_available: Optional[str] = None

    def __str__(self) -> str:
        status = "✓ AVAILABLE" if self.is_available_at_prediction else "✗ NOT AVAILABLE"
        return f"{self.feature_name}: {status} ({self.availability.value}) - {self.reasoning}"


class FeatureValidator:
    """
    Validates temporal availability of features for build prediction.

    This class analyzes whether features can be computed at prediction time
    (before build execution starts) by examining:
    - Feature naming patterns
    - Computation requirements
    - Data dependencies
    - Temporal constraints
    """

    # Features that require build execution to complete
    EXECUTION_REQUIRED = {
        'duration', 'time', 'runtime', 'elapsed', 'latency',
        'coverage', 'tested', 'executed', 'performance',
        'memory', 'cpu', 'resource', 'log', 'output',
        'result', 'status', 'outcome', 'success', 'failed'
    }

    # Features that change over time (not static)
    TIME_DEPENDENT = {
        'stars', 'forks', 'watchers', 'subscribers',
        'downloads', 'views', 'followers', 'popularity',
        'trending', 'recent', 'current', 'latest'
    }

    # Static features that never change
    STATIC_FEATURES = {
        'name', 'id', 'language', 'license', 'created_at',
        'author', 'owner', 'description', 'topics'
    }

    # Pre-build accessible features
    PRE_BUILD_ACCESSIBLE = {
        'previous', 'prior', 'last', 'historical', 'past',
        'commit', 'code', 'file', 'line', 'change',
        'added', 'deleted', 'modified', 'count', 'number'
    }

    def __init__(self, prediction_point: str = "before_build", strict: bool = False):
        """
        Initialize validator.

        Args:
            prediction_point: When prediction is made ('before_build', 'during_build', 'after_build')
            strict: If True, classify ambiguous features as unavailable
        """
        self.prediction_point = prediction_point
        self.strict = strict
        self._cache: Dict[str, AvailabilityReport] = {}

    def validate_feature(self,
                        feature_name: str,
                        metadata: Optional[Dict] = None) -> AvailabilityReport:
        """
        Validate temporal availability of a single feature.

        Args:
            feature_name: Name of feature to validate
            metadata: Optional metadata about feature (description, source, etc.)

        Returns:
            AvailabilityReport with detailed analysis
        """
        # Check cache
        if feature_name in self._cache:
            return self._cache[feature_name]

        feature_lower = feature_name.lower()
        dependencies = []
        availability = TemporalAvailability.UNKNOWN
        reasoning = "No specific pattern detected"

        # Strategy 1: Check if requires execution
        if any(keyword in feature_lower for keyword in self.EXECUTION_REQUIRED):
            availability = TemporalAvailability.AFTER_BUILD
            reasoning = f"Requires build execution: contains '{[k for k in self.EXECUTION_REQUIRED if k in feature_lower]}'"

        # Strategy 2: Check if time-dependent
        elif any(keyword in feature_lower for keyword in self.TIME_DEPENDENT):
            availability = TemporalAvailability.TIME_DEPENDENT
            reasoning = f"Time-dependent: contains '{[k for k in self.TIME_DEPENDENT if k in feature_lower]}'"

        # Strategy 3: Check if static
        elif any(keyword in feature_lower for keyword in self.STATIC_FEATURES):
            availability = TemporalAvailability.ALWAYS_AVAILABLE
            reasoning = f"Static metadata: contains '{[k for k in self.STATIC_FEATURES if k in feature_lower]}'"

        # Strategy 4: Check if pre-build accessible
        elif any(keyword in feature_lower for keyword in self.PRE_BUILD_ACCESSIBLE):
            availability = TemporalAvailability.BEFORE_BUILD
            reasoning = f"Pre-build accessible: contains '{[k for k in self.PRE_BUILD_ACCESSIBLE if k in feature_lower]}'"

        # Strategy 5: Use metadata if provided
        if metadata and availability == TemporalAvailability.UNKNOWN:
            if 'availability' in metadata:
                avail_str = metadata['availability'].lower()
                if 'before' in avail_str or 'pre-build' in avail_str:
                    availability = TemporalAvailability.BEFORE_BUILD
                    reasoning = f"Metadata indicates: {metadata['availability']}"
                elif 'after' in avail_str or 'post-build' in avail_str:
                    availability = TemporalAvailability.AFTER_BUILD
                    reasoning = f"Metadata indicates: {metadata['availability']}"

            if 'depends_on' in metadata:
                dependencies = metadata['depends_on'] if isinstance(metadata['depends_on'], list) else [metadata['depends_on']]

        # Determine if available at prediction point
        is_available = self._is_available_at_point(availability)

        # Apply strict mode
        if self.strict and availability == TemporalAvailability.UNKNOWN:
            is_available = False
            reasoning = "Ambiguous classification (strict mode: unavailable)"

        report = AvailabilityReport(
            feature_name=feature_name,
            availability=availability,
            is_available_at_prediction=is_available,
            reasoning=reasoning,
            dependencies=dependencies,
            computation_time=self._get_computation_time(availability),
            earliest_available=self._get_earliest_available(availability)
        )

        self._cache[feature_name] = report
        return report

    def validate_features(self,
                         feature_names: List[str],
                         metadata: Optional[Dict[str, Dict]] = None) -> Dict[str, AvailabilityReport]:
        """
        Validate multiple features.

        Args:
            feature_names: List of feature names
            metadata: Optional dict mapping feature names to metadata

        Returns:
            Dictionary mapping feature names to AvailabilityReports
        """
        if metadata is None:
            metadata = {}

        return {
            name: self.validate_feature(name, metadata.get(name, {}))
            for name in feature_names
        }

    def get_available_features(self,
                              feature_names: List[str],
                              metadata: Optional[Dict[str, Dict]] = None) -> List[str]:
        """
        Get list of features available at prediction point.

        Args:
            feature_names: List of feature names to check
            metadata: Optional feature metadata

        Returns:
            List of feature names available at prediction point
        """
        reports = self.validate_features(feature_names, metadata)
        return [name for name, report in reports.items()
                if report.is_available_at_prediction]

    def get_unavailable_features(self,
                                feature_names: List[str],
                                metadata: Optional[Dict[str, Dict]] = None) -> List[str]:
        """
        Get list of features NOT available at prediction point.

        Args:
            feature_names: List of feature names to check
            metadata: Optional feature metadata

        Returns:
            List of feature names not available at prediction point
        """
        reports = self.validate_features(feature_names, metadata)
        return [name for name, report in reports.items()
                if not report.is_available_at_prediction]

    def _is_available_at_point(self, availability: TemporalAvailability) -> bool:
        """Determine if feature is available at current prediction point"""
        if availability == TemporalAvailability.ALWAYS_AVAILABLE:
            return True

        if self.prediction_point == "before_build":
            return availability in [
                TemporalAvailability.BEFORE_BUILD,
                TemporalAvailability.ALWAYS_AVAILABLE
            ]
        elif self.prediction_point == "during_build":
            return availability in [
                TemporalAvailability.BEFORE_BUILD,
                TemporalAvailability.DURING_BUILD,
                TemporalAvailability.ALWAYS_AVAILABLE
            ]
        elif self.prediction_point == "after_build":
            return True  # Everything is available after build
        else:
            return False

    def _get_computation_time(self, availability: TemporalAvailability) -> Optional[str]:
        """Get when feature can be computed"""
        mapping = {
            TemporalAvailability.BEFORE_BUILD: "Pre-build (commit time)",
            TemporalAvailability.DURING_BUILD: "During build execution",
            TemporalAvailability.AFTER_BUILD: "Post-build (completion time)",
            TemporalAvailability.TIME_DEPENDENT: "Variable (changes over time)",
            TemporalAvailability.ALWAYS_AVAILABLE: "Any time (static)",
            TemporalAvailability.UNKNOWN: "Unknown"
        }
        return mapping.get(availability)

    def _get_earliest_available(self, availability: TemporalAvailability) -> Optional[str]:
        """Get earliest point when feature becomes available"""
        mapping = {
            TemporalAvailability.BEFORE_BUILD: "Commit time",
            TemporalAvailability.DURING_BUILD: "Build start",
            TemporalAvailability.AFTER_BUILD: "Build completion",
            TemporalAvailability.TIME_DEPENDENT: "N/A (dynamic)",
            TemporalAvailability.ALWAYS_AVAILABLE: "Repository creation",
            TemporalAvailability.UNKNOWN: "Unknown"
        }
        return mapping.get(availability)

    def print_report(self, reports: Dict[str, AvailabilityReport]):
        """Print formatted availability report"""
        available = [r for r in reports.values() if r.is_available_at_prediction]
        unavailable = [r for r in reports.values() if not r.is_available_at_prediction]

        print(f"\n{'='*80}")
        print(f"TEMPORAL AVAILABILITY REPORT (Prediction Point: {self.prediction_point})")
        print(f"{'='*80}\n")

        print(f"Total features: {len(reports)}")
        print(f"  ✓ Available:   {len(available)} ({len(available)/len(reports)*100:.1f}%)")
        print(f"  ✗ Unavailable: {len(unavailable)} ({len(unavailable)/len(reports)*100:.1f}%)\n")

        if available:
            print(f"\nAVAILABLE FEATURES ({len(available)}):")
            print("-" * 80)
            for report in sorted(available, key=lambda r: r.feature_name):
                print(f"  ✓ {report.feature_name:<40} {report.availability.value}")

        if unavailable:
            print(f"\nUNAVAILABLE FEATURES ({len(unavailable)}):")
            print("-" * 80)
            for report in sorted(unavailable, key=lambda r: r.feature_name):
                print(f"  ✗ {report.feature_name:<40} {report.availability.value}")
                print(f"     Reason: {report.reasoning}")

        print(f"\n{'='*80}\n")

    def export_report(self,
                     reports: Dict[str, AvailabilityReport],
                     output_path: str,
                     format: str = 'csv'):
        """Export availability reports to file"""
        data = []
        for name, report in reports.items():
            data.append({
                'feature': name,
                'availability': report.availability.value,
                'available_at_prediction': report.is_available_at_prediction,
                'reasoning': report.reasoning,
                'computation_time': report.computation_time,
                'earliest_available': report.earliest_available,
                'dependencies': ','.join(report.dependencies) if report.dependencies else ''
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

        print(f"✓ Availability report exported to: {output_path}")

    def clear_cache(self):
        """Clear the validation cache"""
        self._cache.clear()
