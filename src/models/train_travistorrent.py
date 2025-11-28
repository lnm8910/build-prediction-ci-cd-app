#!/usr/bin/env python3
"""
Train with ONLY pre-build features (no outcome-dependent data)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

def get_pre_build_features():
    """
    Return only features that are available BEFORE build execution
    (No outcomes, no build results, no test results)
    """
    return [
        # Code metrics (available from git/GitHub before build)
        'git_diff_src_churn',
        'gh_diff_files_added',
        'gh_diff_files_deleted',
        'gh_diff_files_modified',
        'gh_diff_src_files',
        'gh_diff_doc_files',
        'gh_diff_other_files',
        'gh_sloc',
        'gh_num_commits_on_files_touched',
        'code_churn_rate',
        'files_changed_total',
        'commit_frequency',

        # Build context (NOT build results)
        'tr_build_number',  # Build sequence number

        # Test structure (NOT test results)
        'gh_diff_tests_added',
        'gh_diff_tests_deleted',
        'gh_test_lines_per_kloc',
        'gh_test_cases_per_kloc',
        'gh_asserts_cases_per_kloc',

        # Team/collaboration
        'gh_team_size',
        'gh_num_commit_comments',
        'total_comments',
        'comment_density',
        'core_team_contribution',

        # Project context
        'gh_project_name',
        'gh_lang',
        'git_branch',
        'gh_repo_age',
        'gh_repo_num_commits',
        'project_maturity_days',
        'commit_volume',

        # Time-based
        'build_hour',
        'build_day_of_week',
        'build_month',
        'build_is_weekend',
    ]

def main():
    print("="*80)
    print("CLEAN FEATURE TRAINING (Pre-Build Features Only)")
    print("="*80)

    # Load data
    print("\n📂 Loading data...")
    df_features = pd.read_csv('./travis_sdlc_export/combined_features.csv')
    df_targets = pd.read_csv('./travis_sdlc_export/targets.csv')

    # Select ONLY pre-build features
    pre_build_cols = get_pre_build_features()
    available_cols = [col for col in pre_build_cols if col in df_features.columns]

    print(f"\n✓ Total pre-build features requested: {len(pre_build_cols)}")
    print(f"✓ Available in dataset: {len(available_cols)}")
    print(f"✓ Missing: {set(pre_build_cols) - set(available_cols)}")

    X = df_features[available_cols].copy()
    y = df_targets['build_success']

    print(f"\n📊 Dataset:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Build success rate: {y.mean():.2%}")

    # Handle categoricals
    print(f"\n🔧 Preprocessing...")
    categorical_cols = X.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col].astype(str))
        print(f"  ✓ Encoded: {col}")

    # Handle missing
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    print(f"  ✓ Handled missing values")

    # Remove constant columns
    variance = X.var()
    non_constant = variance[variance > 0].index
    X = X[non_constant]
    print(f"  ✓ Active features after removing constants: {X.shape[1]}")

    # Split
    print(f"\n✂️ Train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  ✓ Train: {len(X_train):,}")
    print(f"  ✓ Test: {len(X_test):,}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    print(f"\n🎯 Training models...")
    print("="*80)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    }

    results = []

    for name, model in models.items():
        print(f"\n📊 {name}:")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0

        print(f"  ✓ Accuracy:  {acc:.4f}")
        print(f"  ✓ Precision: {prec:.4f}")
        print(f"  ✓ Recall:    {rec:.4f}")
        print(f"  ✓ F1 Score:  {f1:.4f}")
        if y_pred_proba is not None:
            print(f"  ✓ ROC AUC:   {auc:.4f}")

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC-AUC': auc
        })

    # Results table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY (Pre-Build Features Only)")
    print(f"{'='*80}")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Best model
    best_idx = results_df['Accuracy'].idxmax()
    best_name = results_df.loc[best_idx, 'Model']
    best_model = models[best_name]

    print(f"\n🏆 Best Model: {best_name}")
    print(f"   Accuracy: {results_df.loc[best_idx, 'Accuracy']:.2%}")

    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]

        print(f"\n📊 Top 15 Features ({best_name}):")
        print("="*80)
        for i, idx in enumerate(indices, 1):
            print(f"{i:2d}. {X.columns[idx]:35s} {importances[idx]:.4f}")

    # Classification report
    print(f"\n📋 Detailed Report ({best_name}):")
    print("="*80)
    y_pred_best = best_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_best, target_names=['Failed', 'Passed']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_best)
    print(f"\n📊 Confusion Matrix:")
    print(f"                Predicted")
    print(f"                Failed  Passed")
    print(f"Actual Failed   {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"       Passed   {cm[1,0]:6d}  {cm[1,1]:6d}")

    # Save
    joblib.dump(best_model, 'clean_model.pkl')
    joblib.dump(scaler, 'clean_scaler.pkl')
    results_df.to_csv('clean_model_results.csv', index=False)

    print(f"\n💾 Saved:")
    print(f"  ✓ clean_model.pkl")
    print(f"  ✓ clean_scaler.pkl")
    print(f"  ✓ clean_model_results.csv")

    print(f"\n✅ Training complete with realistic accuracy!")
    print(f"   (Using only features available BEFORE build execution)")

if __name__ == "__main__":
    main()
