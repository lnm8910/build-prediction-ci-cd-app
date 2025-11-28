#!/usr/bin/env python3
"""
Train ML Models on GHALogs Enriched Dataset
Tests both clean and leaky models to measure data leakage impact
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import json
from datetime import datetime

# Parse arguments
parser = argparse.ArgumentParser(description='Train ML models on GHALogs')
parser.add_argument('--model', type=str, default='rf', choices=['rf', 'gb', 'lr', 'dt', 'all'])
args = parser.parse_args()

print("=" * 80)
print("GHALOGS ML TRAINING - CLEAN VS LEAKY MODEL COMPARISON")
print("=" * 80)

# Load datasets
print("\n[1/8] Loading datasets...")
df_clean = pd.read_csv('output/ghalogs_clean.csv')
df_leaky = pd.read_csv('output/ghalogs_enriched.csv')

print(f"   ✓ Clean dataset: {len(df_clean):,} rows × {len(df_clean.columns)} features")
print(f"   ✓ Leaky dataset: {len(df_leaky):,} rows × {len(df_leaky.columns)} features")

# Prepare clean dataset
print("\n[2/8] Preparing clean dataset (no data leakage)...")
clean_exclude = ['repo_name', 'head_sha', 'conclusion', 'created_at', 'commit_author', 'repo_default_branch']
clean_features = [col for col in df_clean.columns if col not in clean_exclude]

X_clean = df_clean[clean_features].copy()
y_clean = (df_clean['conclusion'] == 'success').astype(int)

# Handle categorical features (one-hot encode)
categorical_cols = X_clean.select_dtypes(include=['object']).columns.tolist()
print(f"   ✓ Categorical features to encode: {categorical_cols}")

X_clean = pd.get_dummies(X_clean, columns=categorical_cols, drop_first=True)

# Handle missing values
for col in X_clean.columns:
    if X_clean[col].dtype in ['float64', 'int64']:
        X_clean[col] = X_clean[col].fillna(X_clean[col].median())

print(f"   ✓ Clean features: {len(clean_features)}")
print(f"   ✓ Feature list: {clean_features}")
print(f"   ✓ Target distribution: {y_clean.value_counts().to_dict()}")

# Prepare leaky dataset
print("\n[3/8] Preparing leaky dataset (with time-dependent features)...")
leaky_exclude = ['repo_name', 'head_sha', 'conclusion', 'created_at', 'commit_author', 'repo_default_branch']
leaky_features = [col for col in df_leaky.columns if col not in leaky_exclude]

X_leaky = df_leaky[leaky_features].copy()
y_leaky = (df_leaky['conclusion'] == 'success').astype(int)

# Handle categorical features
categorical_cols_leaky = X_leaky.select_dtypes(include=['object']).columns.tolist()
X_leaky = pd.get_dummies(X_leaky, columns=categorical_cols_leaky, drop_first=True)

for col in X_leaky.columns:
    if X_leaky[col].dtype in ['float64', 'int64']:
        X_leaky[col] = X_leaky[col].fillna(X_leaky[col].median())

print(f"   ✓ Leaky features: {len(leaky_features)}")
print(f"   ✓ Additional features vs clean: {set(leaky_features) - set(clean_features)}")

# Train/test split (80/20, stratified)
print("\n[4/8] Creating train/test splits (80/20, stratified)...")
X_clean_train, X_clean_test, y_clean_train, y_clean_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
)

X_leaky_train, X_leaky_test, y_leaky_train, y_leaky_test = train_test_split(
    X_leaky, y_leaky, test_size=0.2, random_state=42, stratify=y_leaky
)

print(f"   ✓ Clean train: {len(X_clean_train):,} samples")
print(f"   ✓ Clean test:  {len(X_clean_test):,} samples")

# Feature scaling
print("\n[5/8] Scaling features...")
scaler_clean = StandardScaler()
X_clean_train_scaled = scaler_clean.fit_transform(X_clean_train)
X_clean_test_scaled = scaler_clean.transform(X_clean_test)

scaler_leaky = StandardScaler()
X_leaky_train_scaled = scaler_leaky.fit_transform(X_leaky_train)
X_leaky_test_scaled = scaler_leaky.transform(X_leaky_test)

print(f"   ✓ Features scaled with StandardScaler")

# Train clean model
print("\n[6/8] Training CLEAN model (RandomForest)...")
rf_clean = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf_clean.fit(X_clean_train_scaled, y_clean_train)
print(f"   ✓ Clean RandomForest trained")

# Evaluate clean model
y_clean_pred = rf_clean.predict(X_clean_test_scaled)
y_clean_proba = rf_clean.predict_proba(X_clean_test_scaled)[:, 1]

clean_results = {
    'model': 'RandomForest (Clean)',
    'features': len(clean_features),
    'accuracy': accuracy_score(y_clean_test, y_clean_pred),
    'precision': precision_score(y_clean_test, y_clean_pred),
    'recall': recall_score(y_clean_test, y_clean_pred),
    'f1': f1_score(y_clean_test, y_clean_pred),
    'roc_auc': roc_auc_score(y_clean_test, y_clean_proba),
    'train_samples': len(X_clean_train),
    'test_samples': len(X_clean_test)
}

print(f"\n   CLEAN MODEL RESULTS:")
print(f"   Accuracy:  {clean_results['accuracy']:.4f} ({clean_results['accuracy']*100:.2f}%)")
print(f"   Precision: {clean_results['precision']:.4f}")
print(f"   Recall:    {clean_results['recall']:.4f}")
print(f"   F1 Score:  {clean_results['f1']:.4f}")
print(f"   ROC-AUC:   {clean_results['roc_auc']:.4f}")

# Train leaky model
print("\n[7/8] Training LEAKY model (RandomForest)...")
rf_leaky = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf_leaky.fit(X_leaky_train_scaled, y_leaky_train)
print(f"   ✓ Leaky RandomForest trained")

# Evaluate leaky model
y_leaky_pred = rf_leaky.predict(X_leaky_test_scaled)
y_leaky_proba = rf_leaky.predict_proba(X_leaky_test_scaled)[:, 1]

leaky_results = {
    'model': 'RandomForest (Leaky)',
    'features': len(leaky_features),
    'accuracy': accuracy_score(y_leaky_test, y_leaky_pred),
    'precision': precision_score(y_leaky_test, y_leaky_pred),
    'recall': recall_score(y_leaky_test, y_leaky_pred),
    'f1': f1_score(y_leaky_test, y_leaky_pred),
    'roc_auc': roc_auc_score(y_leaky_test, y_leaky_proba),
    'train_samples': len(X_leaky_train),
    'test_samples': len(X_leaky_test)
}

print(f"\n   LEAKY MODEL RESULTS:")
print(f"   Accuracy:  {leaky_results['accuracy']:.4f} ({leaky_results['accuracy']*100:.2f}%)")
print(f"   Precision: {leaky_results['precision']:.4f}")
print(f"   Recall:    {leaky_results['recall']:.4f}")
print(f"   F1 Score:  {leaky_results['f1']:.4f}")
print(f"   ROC-AUC:   {leaky_results['roc_auc']:.4f}")

# Calculate leakage tax
print("\n[8/8] Calculating data leakage impact...")
leakage_tax = {
    'accuracy_tax': (leaky_results['accuracy'] - clean_results['accuracy']) * 100,
    'roc_auc_tax': (leaky_results['roc_auc'] - clean_results['roc_auc']) * 100,
    'clean_accuracy': clean_results['accuracy'],
    'leaky_accuracy': leaky_results['accuracy'],
    'clean_roc_auc': clean_results['roc_auc'],
    'leaky_roc_auc': leaky_results['roc_auc']
}

print(f"\n   DATA LEAKAGE IMPACT:")
print(f"   Accuracy:  {clean_results['accuracy']*100:.2f}% → {leaky_results['accuracy']*100:.2f}% (+{leakage_tax['accuracy_tax']:.2f}pp)")
print(f"   ROC-AUC:   {clean_results['roc_auc']*100:.2f}% → {leaky_results['roc_auc']*100:.2f}% (+{leakage_tax['roc_auc_tax']:.2f}pp)")

# Feature importance (clean model)
feature_importance = pd.DataFrame({
    'feature': X_clean_train.columns,
    'importance': rf_clean.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   TOP 10 FEATURES (Clean Model):")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:<30} {row['importance']:.4f}")

# Save results
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save models
joblib.dump(rf_clean, 'output/ghalogs_clean_model.pkl')
joblib.dump(scaler_clean, 'output/ghalogs_clean_scaler.pkl')
joblib.dump(rf_leaky, 'output/ghalogs_leaky_model.pkl')
joblib.dump(scaler_leaky, 'output/ghalogs_leaky_scaler.pkl')
print("   ✓ Models saved:")
print("     - output/ghalogs_clean_model.pkl")
print("     - output/ghalogs_clean_scaler.pkl")
print("     - output/ghalogs_leaky_model.pkl")
print("     - output/ghalogs_leaky_scaler.pkl")

# Save results
results_data = {
    'timestamp': datetime.now().isoformat(),
    'dataset': {
        'total_runs': len(df_clean),
        'train_samples': len(X_clean_train),
        'test_samples': len(X_clean_test),
        'success_rate': (y_clean == 1).sum() / len(y_clean),
        'failure_rate': (y_clean == 0).sum() / len(y_clean)
    },
    'clean_model': clean_results,
    'leaky_model': leaky_results,
    'leakage_tax': leakage_tax,
    'top_features': feature_importance.head(10).to_dict('records')
}

with open('output/ghalogs_results.json', 'w') as f:
    json.dump(results_data, f, indent=2)
print("   ✓ Results saved: output/ghalogs_results.json")

# Save feature importance
feature_importance.to_csv('output/ghalogs_feature_importance.csv', index=False)
print("   ✓ Feature importance: output/ghalogs_feature_importance.csv")

# Print final summary
print("\n" + "=" * 80)
print("✅ ML TRAINING COMPLETE!")
print("=" * 80)
print(f"\n🎯 CLEAN MODEL PERFORMANCE:")
print(f"   Accuracy:  {clean_results['accuracy']*100:.2f}%")
print(f"   ROC-AUC:   {clean_results['roc_auc']*100:.2f}%")
print(f"   Precision: {clean_results['precision']*100:.2f}%")
print(f"   Recall:    {clean_results['recall']*100:.2f}%")
print(f"   F1 Score:  {clean_results['f1']*100:.2f}%")

print(f"\n📊 DATA LEAKAGE TAX:")
print(f"   Accuracy increase: +{leakage_tax['accuracy_tax']:.2f} percentage points")
print(f"   ROC-AUC increase:  +{leakage_tax['roc_auc_tax']:.2f} percentage points")

print(f"\n🏆 TOP 3 FEATURES:")
for idx, row in feature_importance.head(3).iterrows():
    print(f"   {idx+1}. {row['feature']:<30} ({row['importance']*100:.2f}%)")

# Comparison to TravisTorrent baseline
print(f"\n📈 COMPARISON TO TRAVISTORRENT:")
print(f"   TravisTorrent (clean): 82.73% accuracy")
print(f"   GHALogs 5K (clean):    {clean_results['accuracy']*100:.2f}% accuracy")
diff = (clean_results['accuracy'] - 0.8273) * 100
status = "✅ Better" if diff > 0 else "⚠️ Lower"
print(f"   Difference: {diff:+.2f}pp {status}")

print("\n" + "=" * 80)
print("Next steps:")
print("1. Review results in output/ghalogs_5k_results.json")
print("2. Check feature importance rankings")
print("3. If satisfied, scale to full 86K enrichment (18-22 hours)")
print("=" * 80)
