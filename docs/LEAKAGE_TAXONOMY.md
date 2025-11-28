# Temporal Data Leakage Taxonomy
## A Systematic 3-Type Classification System

**Document Version:** 1.0
**Last Updated:** November 2025
**Authors:** Amit Rangari, Lalit Narayan Mishra, Sandesh Nagrare, Saroj Kumar Nayak

---

## Overview

This document describes the **3-type temporal data leakage taxonomy** for detecting and preventing temporal leakage in machine learning-based software engineering prediction tasks.

Temporal data leakage occurs when features used for prediction incorporate information that would not be available at prediction time in a real-world deployment scenario, causing inflated performance estimates that don't generalize to prospective prediction.

**Our taxonomy identifies three distinct types of temporal leakage:**

1. **Type 1: Direct Outcome Encoding** - Features that explicitly encode the outcome being predicted
2. **Type 2: Execution-Dependent Metrics** - Features computable only after the predicted event occurs
3. **Type 3: Future Information Leakage** - Features incorporating data from after the prediction timestamp

---

## Type 1: Direct Outcome Encoding

### Definition
Features that **explicitly encode the outcome** being predicted, either directly or through strongly correlated proxies.

### Detection Criteria
- Feature name contains outcome-related keywords: `status`, `result`, `outcome`, `success`, `failed`, `error`
- Perfect or near-perfect correlation with target (|r| > 0.9)
- Feature values directly map to outcome classes

### TravisTorrent Examples (5 features removed)

| Feature | Description | Why It's Leaky |
|---------|-------------|----------------|
| `tr_status` | Build status (passed/errored/failed) | **This IS the outcome** we're predicting |
| `tr_log_status` | Build status from logs | Duplicate encoding of outcome |
| `build_success` | Binary build outcome (0/1) | **The prediction target itself** |
| `tr_log_bool_tests_failed` | Boolean: any tests failed | Direct indicator of test failure outcome |
| `tr_log_num_tests_failed` | Count of failed tests | Direct encoding of test failures |
| `tr_log_num_test_suites_failed` | Failing test suite count | Direct encoding of suite failures |
| `test_suite_failure_rate` | Percentage of failing suites | Computed directly from failure counts |

### GHALogs Examples (1 feature removed)

| Feature | Description | Why It's Leaky |
|---------|-------------|----------------|
| `conclusion` | Workflow outcome (success/failure) | **This IS the prediction target** |

### Impact on TravisTorrent
- **Without Type 1 features:** Model can make predictions
- **With Type 1 features:** 99.99% accuracy (perfect hindsight, zero prospective utility)
- **Detection:** Correlation analysis (|r| > 0.95 with target)

### Analogy
Using `tr_status` to predict build success is like using tomorrow's stock closing price to predict whether the stock will go up today - technically accurate in retrospect but impossible to deploy prospectively.

---

## Type 2: Execution-Dependent Metrics

### Definition
Features that can **only be computed after** the predicted event has occurred and completed execution.

### Detection Criteria
- Requires build/test execution to complete before feature can be computed
- Feature name contains execution keywords: `duration`, `time`, `runtime`, `coverage`, `executed`, `log`
- Value depends on runtime behavior, resource consumption, or execution results

### TravisTorrent Examples (7 features removed)

| Feature | Description | Why It's Leaky | Availability |
|---------|-------------|----------------|--------------|
| `tr_duration` | Build execution duration (seconds) | Need to run build to measure | After build completes |
| `tr_log_buildduration` | Build duration from logs | Extracted from completed logs | After build completes |
| `tr_log_setup_time` | Setup phase duration | Only measurable after setup runs | After setup completes |
| `build_time_minutes` | Build time in minutes | Derived from tr_duration | After build completes |
| `setup_time_minutes` | Setup time in minutes | Derived from setup duration | After setup completes |
| `build_efficiency` | Efficiency score | Requires duration measurements | After build completes |
| `tr_log_num_tests_run` | Total tests executed | Only known after test execution | After tests complete |
| `tr_log_num_tests_skipped` | Skipped test count | Test execution statistics | After tests complete |
| `tr_log_num_test_suites_run` | Test suites executed | Suite execution count | After tests complete |
| `test_success_rate` | Percentage of passing tests | Computed from test results | After tests complete |
| `test_coverage_delta` | Change in code coverage | Requires coverage analysis | After tests complete |
| `quality_score` | Composite quality metric | Aggregates build + test results | After all phases complete |

### GHALogs Examples (0 features removed)
- **None detected** - GHALogs dataset primarily contains metadata, not execution metrics
- Workflow duration not included in enriched dataset

### Impact on TravisTorrent
- **Clean accuracy:** 82.73% (realistic, deployable)
- **With Type 2 features:** 91.45% accuracy (+8.72pp inflation)
- **Reason:** Duration and test counts highly correlate with success but unavailable pre-build

### Real-World Constraint
In a production CI/CD system making predictions **before** build execution, execution-dependent metrics are not yet available. Including them produces retrospective accuracy but zero prospective utility.

### Analogy
Using build duration to predict build success is like using a marathon runner's finishing time to predict whether they'll complete the race - accurate in retrospect but impossible to know beforehand.

---

## Type 3: Future Information Leakage

### Definition
Features that **incorporate information from after** the prediction timestamp, typically time-dependent metrics that change over time.

### Detection Criteria
- Feature values change after the prediction event occurs
- Time-dependent popularity metrics: `stars`, `forks`, `watchers`, `downloads`
- Rolling averages or cumulative metrics that inadvertently include future data
- Timestamp analysis reveals future dependency

### TravisTorrent Examples (3 features removed)

| Feature | Description | Why It's Leaky | Time Behavior |
|---------|-------------|----------------|---------------|
| `gh_total_stars` | Repository star count | Stars accumulate AFTER build timestamp | Increases over time |
| `gh_total_forks` | Repository fork count | Forks occur AFTER build timestamp | Increases over time |
| `gh_watchers_count` | Repository watcher count | Watchers change AFTER build | Dynamic over time |

### GHALogs Examples (4 features removed)

| Feature | Description | Why It's Leaky | Time Behavior |
|---------|-------------|----------------|---------------|
| `repo_stars` | Repository star count | Stars accumulate after workflow run | Increases over time |
| `repo_forks` | Repository fork count | Forks occur after workflow run | Increases over time |
| `repo_watchers` | Repository watcher count | Watchers change after run | Dynamic over time |
| `repo_open_issues` | Open issues count | Issues opened/closed after run | Changes over time |

### Platform-Dependent Impact (Key Finding!)

**TravisTorrent (Travis CI, 2013-2017):**
- **Clean accuracy:** 82.73%
- **With Type 3 features:** 97.80% (+15.07pp inflation!)
- **Relative inflation:** 18.2%

**GHALogs (GitHub Actions, 2023):**
- **Clean accuracy:** 83.30%
- **With Type 3 features:** 83.77% (+0.48pp inflation)
- **Relative inflation:** 0.6%

**Divergence:** 14.59 percentage points (ratio 31.4:1, p < 0.001)

### Why Platform Matters

**Travis CI (Third-Party Service):**
- Requires external API calls for repository metadata
- Stars/forks highly predictive of project health
- Substantial dependency on time-dependent signals

**GitHub Actions (Native Integration):**
- Tight coupling with GitHub repository data
- Stars/forks provide minimal additional information beyond static metadata
- Native integration enables prediction from static features alone

### Real-World Constraint
When making a prediction at time T₀, we can only use information available at T₀. Using `repo_stars` measured at time T₁ (where T₁ > T₀) incorporates future information.

### Analogy
Using repository stars measured 6 months after a build to predict that build's outcome is like using a company's stock price from next year to predict this year's quarterly earnings - technically feasible with hindsight data but impossible to deploy prospectively.

---

## Filtering Process

### Step-by-Step Application

#### Step 1: Temporal Availability Audit
Review dataset schema documentation to identify when each feature becomes available:
- **Before build:** Static metadata, historical aggregates, commit-level metrics
- **During build:** Intermediate execution state (rare in datasets)
- **After build:** Duration, test results, coverage, log-based metrics

#### Step 2: Keyword-Based Detection
Scan feature names for leakage indicators:
- **Type 1 keywords:** `status`, `result`, `outcome`, `success`, `failed`, `error`
- **Type 2 keywords:** `duration`, `time`, `runtime`, `coverage`, `executed`, `log`
- **Type 3 keywords:** `stars`, `forks`, `watchers`, `recent`, `current`

#### Step 3: Correlation Analysis
Compute correlation between each feature and target:
- **|r| > 0.9:** Flag as potential Type 1 (direct outcome encoding)
- **0.7 < |r| < 0.9:** Investigate further (may be legitimate or leaky)
- **|r| < 0.7:** Likely safe from direct encoding

#### Step 4: Temporal Validation
For time-dependent features:
- Verify feature value at prediction timestamp T₀
- Check if feature changes between T₀ and T₁ (where T₁ > T₀)
- If changes substantially: **Type 3 leakage detected**

#### Step 5: Manual Review
Subject matter experts review ambiguous cases:
- Features with mixed signals from automated detection
- Domain-specific knowledge required (e.g., is test count static or dynamic?)
- Conservative approach: When uncertain, classify as leaky

---

## Results: Filtering Statistics

### TravisTorrent (Travis CI, 2013-2017)
```
Original features:    66
  - Type 1 removed:    5  (Direct outcome encoding)
  - Type 2 removed:    7  (Execution-dependent)
  - Type 3 removed:    3  (Future information)
  - Total removed:    15  (22.7% of features)

Clean features:       31  (47.0% of original)
  - Code metrics:      9  (29.0%)
  - Project maturity:  8  (25.8%)
  - Test structure:    6  (19.4%)
  - Build history:     5  (16.1%)
  - Commit context:    3  ( 9.7%)
```

### GHALogs (GitHub Actions, 2023)
```
Original features:    33
  - Type 1 removed:    1  (Direct outcome encoding)
  - Type 2 removed:    0  (Execution-dependent - none in dataset)
  - Type 3 removed:    4  (Future information)
  - Total removed:     5  (15.2% of features)

Clean features:       29  (87.9% of original)
  - Project maturity:  7  (24.1%)
  - Build history:     4  (13.8%)
  - Code complexity:  10  (34.5%)
  - Test structure:    1  ( 3.4%)
  - Commit context:    3  (10.3%)
  - Metadata:          4  (13.8%)
```

---

## Quantified Leakage Impact

### TravisTorrent Leakage Tax
| Feature Set | Accuracy | ROC-AUC | F1 Score | Inflation |
|-------------|----------|---------|----------|-----------|
| **31 Clean Features** | **82.73%** | **91.38%** | **89.80%** | **Baseline** |
| 66 All Features (leaky) | 97.80% | 99.56% | 98.42% | +15.07pp |

**Interpretation:** Including time-dependent features inflates accuracy by **15.07 percentage points** (18.2% relative inflation). This performance is **unachievable in prospective prediction**.

### GHALogs Leakage Tax
| Feature Set | Accuracy | ROC-AUC | F1 Score | Inflation |
|-------------|----------|---------|----------|-----------|
| **29 Clean Features** | **83.30%** | **80.10%** | **89.97%** | **Baseline** |
| 33 All Features (leaky) | 83.77% | 82.25% | 90.17% | +0.48pp |

**Interpretation:** Including time-dependent features inflates accuracy by only **0.48 percentage points** (0.6% relative inflation). **Minimal leakage on GitHub Actions!**

### Platform Divergence (Novel Finding)
- **Leakage tax difference:** 14.59 percentage points
- **Ratio:** 31.4:1 (Travis CI vs GitHub Actions)
- **Statistical significance:** p < 0.001
- **Conclusion:** **Platform architecture fundamentally affects metadata predictiveness**

---

## Usage Examples

### Using the Automated Toolkit

```python
from src.leakage_detection import TemporalLeakageTaxonomy, FeatureValidator, CorrelationAnalyzer

# Load your dataset
import pandas as pd
df = pd.read_csv('your_dataset.csv')

# Step 1: Classify features with taxonomy
taxonomy = TemporalLeakageTaxonomy(
    strict_mode=False,
    correlation_threshold=0.9,
    verbose=True
)

classifications = taxonomy.classify_features(
    data=df,
    target_column='build_success',
    timestamp_column='created_at'
)

# Extract clean features
clean_features = taxonomy.get_clean_features(classifications)
print(f"Clean features: {len(clean_features)}")

# Export classification results
taxonomy.export_classification(classifications, 'leakage_taxonomy.csv')

# Step 2: Validate temporal availability
validator = FeatureValidator(prediction_point='before_build', strict=False)
reports = validator.validate_features(df.columns.tolist())
available_features = validator.get_available_features(df.columns.tolist())
validator.print_report(reports)

# Step 3: Analyze correlations
analyzer = CorrelationAnalyzer(threshold=0.9, method='auto', verbose=True)
corr_result = analyzer.analyze(df, target='build_success')
suspicious = analyzer.get_suspicious_features(corr_result)
print(f"Suspicious high-correlation features: {suspicious}")

# Generate correlation plot
analyzer.plot_correlations(corr_result, 'correlations.png', top_n=20)
```

---

## Best Practices

### 1. Always Apply Temporal Validation
Before training any ML model for temporal prediction:
- ✅ List ALL features with their availability timestamps
- ✅ Apply 3-type taxonomy systematically
- ✅ Verify features pass "can we compute this before prediction time?" test
- ✅ Remove ALL features failing temporal validation

### 2. Quantify Leakage Impact
Compare performance with clean vs leaky features:
```python
# Train clean model (legitimate features only)
clean_model = RandomForestClassifier()
clean_model.fit(X_clean_train, y_train)
clean_acc = clean_model.score(X_clean_test, y_test)

# Train leaky model (all features for comparison)
leaky_model = RandomForestClassifier()
leaky_model.fit(X_leaky_train, y_train)
leaky_acc = leaky_model.score(X_leaky_test, y_test)

# Calculate leakage tax
leakage_tax = (leaky_acc - clean_acc) * 100
print(f"Leakage tax: {leakage_tax:.2f} percentage points")
```

### 3. Document Feature Classifications
Create a `features_leakage_classification.csv` file documenting:
- Feature name
- Leakage type (CLEAN, TYPE_1, TYPE_2, TYPE_3)
- Category (Code, Build, Test, Team, Project)
- Availability (BEFORE_BUILD, AFTER_BUILD, TIME_DEPENDENT)
- Reasoning for classification

### 4. Platform-Specific Validation
Different CI/CD platforms have different metadata availability:
- **Travis CI:** Limited native integration, benefits from time-dependent metrics
- **GitHub Actions:** Tight integration, minimal time-dependent leakage
- **GitLab CI:** Similar to GitHub Actions (native integration)
- **Jenkins:** Depends on configuration (may vary)

**Recommendation:** Apply taxonomy to each platform independently and validate which features constitute leakage for that specific architecture.

---

## Common Mistakes to Avoid

### Mistake 1: Using Build Duration for Prediction
❌ **Wrong:**
```python
features = ['code_churn', 'test_count', 'build_duration']  # LEAKY!
model.fit(X[features], y)
```

✅ **Correct:**
```python
features = ['code_churn', 'test_count', 'prev_build_duration']  # OK - historical
model.fit(X[features], y)
```

**Reason:** `build_duration` is only available **after** the build completes (Type 2). Use `prev_build_duration` (historical) instead.

---

### Mistake 2: Using Current Test Results
❌ **Wrong:**
```python
# Using current build's test execution counts
features = ['num_tests_passed', 'num_tests_failed']  # LEAKY!
```

✅ **Correct:**
```python
# Using historical test counts from prior builds
features = ['test_count_static', 'test_density', 'prev_test_success_rate']  # OK
```

**Reason:** Current build's test results encode the outcome directly (Type 1) or require execution (Type 2). Use static test counts from code analysis or historical test patterns.

---

### Mistake 3: Using Time-Dependent Popularity
❌ **Wrong:**
```python
# Using star count measured at any time
features = ['repo_stars', 'repo_forks']  # LEAKY on Travis CI!
```

✅ **Correct:**
```python
# Using snapshot at exact prediction timestamp
features = ['repo_stars_at_commit_time']  # OK - if properly timestamped
# OR better: use static metrics that don't change
features = ['repo_age_days', 'commits_count']  # OK - static
```

**Reason:** Stars/forks change over time. Using current values incorporates future information (Type 3). **Exception:** On GitHub Actions, these provide minimal information beyond static metrics (0.48pp inflation), but still violate temporal constraint.

---

### Mistake 4: Not Quantifying Leakage Impact
❌ **Wrong:**
"Our model achieves 96% accuracy on Travis CI builds!"

✅ **Correct:**
"Our model achieves 82.73% accuracy using only pre-build features. Including execution-dependent features inflates accuracy to 97.80% (+15.07pp), demonstrating that prior 96%+ claims likely suffered from temporal data leakage."

**Reason:** Always report **both** clean and leaky performance to quantify leakage impact and provide realistic expectations.

---

## Validation Checklist

Before claiming prospective predictive utility, verify:

- [ ] **All features pass temporal availability test**
  - Each feature computable at prediction time?
  - No execution-dependent metrics included?
  - No future information incorporated?

- [ ] **Correlation analysis performed**
  - All features have |r| < 0.9 with target?
  - Suspicious correlations investigated?
  - Direct outcome encoding eliminated?

- [ ] **Leakage impact quantified**
  - Trained both clean and leaky models?
  - Calculated leakage tax (accuracy inflation)?
  - Documented inflation in percentage points?

- [ ] **Classification documented**
  - Created features_leakage_taxonomy.csv?
  - Provided reasoning for each classification?
  - Made classification available to reviewers?

- [ ] **Platform-specific validation**
  - Considered platform architecture effects?
  - Validated which features are time-dependent on this platform?
  - Tested generalization to other platforms?

---

## Generalization to Other SE Prediction Tasks

This taxonomy applies beyond build prediction to:

### Defect Prediction
- **Type 1:** Using `defect_count` to predict defects (outcome encoding)
- **Type 2:** Using `execution_coverage` (requires running tests)
- **Type 3:** Using `post-release_downloads` (future information)

### Test Selection
- **Type 1:** Using `test_failed` to predict which tests to run (outcome encoding)
- **Type 2:** Using `test_execution_time` (requires running tests)
- **Type 3:** Using `future_failure_count` (future information)

### Code Review Automation
- **Type 1:** Using `review_verdict` to predict review outcome (outcome encoding)
- **Type 2:** Using `review_duration` (requires review completion)
- **Type 3:** Using `post_merge_issues` (future information)

### General Principle
**The Temporal Availability Test:**
> "Can this feature be computed using **only** information available at prediction time, **before** the event whose outcome we seek to predict?"

If NO → Feature is temporally leaky → Remove from model

---

## Frequently Asked Questions

### Q1: What if I can't remove all Type 2 features?
**A:** Some prediction tasks genuinely require execution. For example:
- **Test prioritization** can use runtime from previous executions (historical Type 2 is OK)
- **Build time prediction** legitimately predicts a Type 2 metric
- **Key:** Use historical execution metrics (previous builds) not current execution metrics

### Q2: Are historical features always safe?
**A:** Mostly yes, with caveats:
- ✅ `prev_build_duration` - OK (historical)
- ✅ `avg_duration_last_10_builds` - OK (historical aggregate)
- ❌ `rolling_avg_including_current` - LEAKY (includes current build)
- **Key:** Historical window must end **before** prediction timestamp

### Q3: How do I handle features with missing temporal metadata?
**A:** Conservative approach:
- **If uncertain:** Classify as leaky and exclude
- **If confident from domain knowledge:** Document reasoning explicitly
- **Best practice:** Request timestamp metadata from data providers

### Q4: What if leakage tax is small (<1pp)?
**A:** Even small leakage violates temporal constraint:
- GHALogs shows 0.48pp tax but still violates prospective prediction principle
- Report both clean and leaky performance for transparency
- Small tax suggests platform has good native integration

### Q5: Can I use this taxonomy for regression tasks?
**A:** Yes! Taxonomy applies to both classification and regression:
- **Classification:** Binary/multi-class outcome prediction
- **Regression:** Continuous value prediction (duration, effort, defect count)
- **Key principle remains:** Only use features available at prediction time

---

## Citation

If you use this taxonomy in your research, please cite:

```bibtex
@article{rangari2025temporal,
  title={A taxonomy for detecting and preventing temporal data leakage in
         machine learning-based build prediction: A dual-platform empirical validation},
  author={Rangari, Amit and Mishra, Lalit Narayan and
          Nagrare, Sandesh and Nayak, Saroj Kumar},
  journal={PLOS ONE},
  year={2025},
  publisher={Public Library of Science},
  doi={10.1371/journal.pone.XXXXXXX}
}
```

---

## References

1. **Kaufman et al.** - "Leakage and the Reproducibility Crisis in ML-based Science" (arXiv:2207.07048)
2. **Kapoor & Narayanan** - "Leakage and the Reproducibility Crisis in Machine Learning-based Science" (Patterns, 2023)
3. **Rangari et al.** - This work (PLOS ONE, 2025)

---

## Contact

Questions about the taxonomy? Contact:

**Lalit Narayan Mishra** (Corresponding Author)
- Email: lnm8910@gmail.com
- Affiliation: Lowe's Companies, Inc.

---

**Document Status:** Final v1.0
**License:** MIT
**Last Updated:** November 2025
