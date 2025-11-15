"""
Retrain Leak Detection Model with Merged Datasets
Combines all available training data and creates production model for Monday demo

This script:
1. Merges all 3 CSV training datasets
2. Extracts 8 statistical features per valve
3. Holds out C402 Cylinder 3 as test set
4. Trains Random Forest on remaining data
5. Evaluates on C402 Cylinder 3 holdout
6. Saves production-ready model artifacts
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import os
from pathlib import Path

print("=" * 80)
print("LEAK DETECTION MODEL RETRAINING - MERGED DATASETS")
print("=" * 80)

# ============================================================================
# 1. LOAD AND MERGE ALL DATASETS
# ============================================================================

print("\n[1/7] Loading datasets...")

# Define data paths
data_dir = Path(__file__).parent.parent / 'data' / 'raw'

# Load all three datasets
df1 = pd.read_csv(data_dir / 'ml_training_dataset.csv')
df2 = pd.read_csv(data_dir / 'ml_training_datasetbatch2.csv')
df3 = pd.read_csv(data_dir / 'fault_tags_training_data.csv')

print(f"  Batch 1: {len(df1)} samples")
print(f"  Batch 2: {len(df2)} samples")
print(f"  Fault Tags: {len(df3)} samples")

# Standardize column names if needed
# Batch 1 & 2 have: Machine ID, Cylinder, Curve, Value, Crank Angle, Fault Classification
# Fault Tags might have different names - check and rename

# Merge datasets
df_merged = pd.concat([df1, df2, df3], ignore_index=True)
print(f"\n  Merged Total: {len(df_merged)} samples")

# ============================================================================
# 2. PREPROCESSING - BINARY CLASSIFICATION
# ============================================================================

print("\n[2/7] Preprocessing for binary classification...")

# Merge label variants for consistency
df_merged['Fault Classification'] = df_merged['Fault Classification'].replace({
    'Leaking valves': 'Valve Leakage',
    'Leak': 'Valve Leakage',
    'leak': 'Valve Leakage'
})

# Filter to binary classification: Leak vs Normal
leak_mask = df_merged['Fault Classification'].str.contains('Leak', case=False, na=False)
normal_mask = df_merged['Fault Classification'].str.contains('Normal', case=False, na=False)

df_binary = df_merged[leak_mask | normal_mask].copy()

# Create binary labels
df_binary['Label'] = df_binary['Fault Classification'].apply(
    lambda x: 1 if 'leak' in x.lower() else 0
)

leak_samples = (df_binary['Label'] == 1).sum()
normal_samples = (df_binary['Label'] == 0).sum()

print(f"  Leak samples: {leak_samples}")
print(f"  Normal samples: {normal_samples}")
print(f"  Total binary samples: {len(df_binary)}")

# ============================================================================
# 3. FEATURE EXTRACTION - GROUP BY VALVE
# ============================================================================

print("\n[3/7] Extracting features per valve...")

# Create Valve_ID: Machine + Cylinder + Curve
df_binary['Valve_ID'] = (
    df_binary['Machine ID'].astype(str) + '_' +
    df_binary['Cylinder'].astype(str) + '_' +
    df_binary['Curve'].astype(str)
)

# Group by Valve_ID and extract aggregate features
valve_features = df_binary.groupby('Valve_ID').agg({
    'Value': ['mean', 'max', 'min', 'std', 'median'],
    'Label': 'first'  # Valve-level label (same for all points)
}).reset_index()

# Flatten column names
valve_features.columns = ['Valve_ID', 'mean_amplitude', 'max_amplitude', 'min_amplitude',
                          'std_amplitude', 'median_amplitude', 'label']

# Calculate crank_angle_at_max separately
crank_angles_at_max = []
for valve_id in valve_features['Valve_ID']:
    valve_data = df_binary[df_binary['Valve_ID'] == valve_id]
    max_idx = valve_data['Value'].idxmax()
    crank_angle = valve_data.loc[max_idx, 'Crank Angle'] if pd.notna(max_idx) else 0
    crank_angles_at_max.append(crank_angle)

valve_features['crank_angle_at_max'] = crank_angles_at_max

# Add derived features
valve_features['amplitude_range'] = (
    valve_features['max_amplitude'] - valve_features['min_amplitude']
)
valve_features['sample_count'] = df_binary.groupby('Valve_ID').size().values

# PHASE 2: Add 5 new leak-specific features
print("  Extracting Phase 2 leak-detection features...")

# Calculate new features for each valve
elevated_percentages = []
mean_to_max_ratios = []
baseline_medians = []
medium_activity_pcts = []
smear_indices = []

for valve_id in valve_features['Valve_ID']:
    valve_data = df_binary[df_binary['Valve_ID'] == valve_id]['Value']

    # Feature 1: Elevated percentage (samples >= 0.5G)
    elevated_count = (valve_data >= 0.5).sum()
    elevated_pct = (elevated_count / len(valve_data)) * 100 if len(valve_data) > 0 else 0
    elevated_percentages.append(elevated_pct)

    # Feature 2: Mean-to-max ratio
    mean_val = valve_data.mean()
    max_val = valve_data.max()
    mean_to_max = mean_val / max_val if max_val > 0 else 0
    mean_to_max_ratios.append(mean_to_max)

    # Feature 3: Baseline median
    baseline_medians.append(valve_data.median())

    # Feature 4: Medium activity percentage (0.5G - 5.0G range)
    medium_count = ((valve_data >= 0.5) & (valve_data < 5.0)).sum()
    medium_pct = (medium_count / len(valve_data)) * 100 if len(valve_data) > 0 else 0
    medium_activity_pcts.append(medium_pct)

    # Feature 5: Smear index
    std_val = valve_data.std()
    median_val = valve_data.median()
    if max_val > 0 and mean_val > 0:
        smear_idx = (std_val / mean_val) * (median_val / max_val)
    else:
        smear_idx = 0
    smear_indices.append(smear_idx)

valve_features['elevated_percentage'] = elevated_percentages
valve_features['mean_to_max_ratio'] = mean_to_max_ratios
valve_features['baseline_median'] = baseline_medians
valve_features['medium_activity_pct'] = medium_activity_pcts
valve_features['smear_index'] = smear_indices

print(f"    Added 5 new features: elevated_percentage, mean_to_max_ratio, baseline_median,")
print(f"                          medium_activity_pct, smear_index")

# PHASE 3: Add 4 pattern detection features (smear vs spike)
print("  Extracting Phase 3 pattern detection features (smear vs spike)...")

continuity_ratios = []
spike_concentrations = []
baseline_elevations = []
iqr_scores = []

for valve_id in valve_features['Valve_ID']:
    valve_data = df_binary[df_binary['Valve_ID'] == valve_id]['Value']

    median_val = valve_data.median()
    max_val = valve_data.max()

    # Feature 6: Continuity ratio - % above median
    above_median = (valve_data > median_val).sum()
    continuity = above_median / len(valve_data) if len(valve_data) > 0 else 0
    continuity_ratios.append(continuity)

    # Feature 7: Spike concentration - % near maximum (>80% of max)
    near_max_threshold = max_val * 0.8
    near_max = (valve_data >= near_max_threshold).sum()
    spike_conc = near_max / len(valve_data) if len(valve_data) > 0 else 0
    spike_concentrations.append(spike_conc)

    # Feature 8: Baseline elevation - 25th percentile / max
    q25 = valve_data.quantile(0.25)
    baseline_elev = q25 / max_val if max_val > 0 else 0
    baseline_elevations.append(baseline_elev)

    # Feature 9: IQR score - inter-quartile range / max
    q75 = valve_data.quantile(0.75)
    iqr = q75 - q25
    iqr_sc = iqr / max_val if max_val > 0 else 0
    iqr_scores.append(iqr_sc)

valve_features['continuity_ratio'] = continuity_ratios
valve_features['spike_concentration'] = spike_concentrations
valve_features['baseline_elevation'] = baseline_elevations
valve_features['iqr_score'] = iqr_scores

print(f"    Added 4 new features: continuity_ratio, spike_concentration,")
print(f"                          baseline_elevation, iqr_score")

# Extract features and labels (now 17 features total)
feature_columns = [
    # Original 8 features
    'mean_amplitude', 'max_amplitude', 'min_amplitude', 'std_amplitude',
    'amplitude_range', 'median_amplitude', 'crank_angle_at_max', 'sample_count',
    # Phase 2: 5 leak-detection features
    'elevated_percentage', 'mean_to_max_ratio', 'baseline_median',
    'medium_activity_pct', 'smear_index',
    # Phase 3: 4 pattern detection features (smear vs spike)
    'continuity_ratio', 'spike_concentration', 'baseline_elevation', 'iqr_score'
]

X = valve_features[feature_columns].values
y = valve_features['label'].values
valve_ids = valve_features['Valve_ID'].values

unique_leak_valves = (y == 1).sum()
unique_normal_valves = (y == 0).sum()

print(f"  Total unique valves: {len(valve_features)}")
print(f"  Unique LEAK valves: {unique_leak_valves}")
print(f"  Unique NORMAL valves: {unique_normal_valves}")
print(f"  Improvement: {unique_leak_valves} leak valves (vs 7 in original model)")

# ============================================================================
# 4. TRAIN/TEST SPLIT - HOLD OUT C402 CYLINDER 3
# ============================================================================

print("\n[4/7] Creating train/test split (C402 Cylinder 3 holdout)...")

# Identify C402 Cylinder 3 samples
c402_cyl3_mask = valve_ids == 'C402-C_3_C402 - C.3CS1.ULTRASONIC G 36KHZ - 44KHZ (NARROW BAND).3CS1'

# Alternative: check if any valve_id contains 'C402' and '3'
if not c402_cyl3_mask.any():
    # Broader match for any C402 Cylinder 3 variant
    c402_cyl3_mask = np.array([('C402' in vid or 'c402' in vid.lower()) and
                                ('.3' in vid or '_3' in vid or '_3_' in vid or
                                 'Cylinder 3' in vid or 'CYL 3' in vid.upper())
                               for vid in valve_ids])

test_indices = np.where(c402_cyl3_mask)[0]
train_indices = np.where(~c402_cyl3_mask)[0]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

print(f"  Training set: {len(X_train)} valves")
print(f"    - Leak: {(y_train == 1).sum()}")
print(f"    - Normal: {(y_train == 0).sum()}")
print(f"  Test set (C402 Cyl 3): {len(X_test)} valves")
if len(X_test) > 0:
    print(f"    - Leak: {(y_test == 1).sum()}")
    print(f"    - Normal: {(y_test == 0).sum()}")
else:
    print("  WARNING: No C402 Cylinder 3 samples found for test set")
    print("  Using 20% random split instead...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {len(X_train)} valves ({(y_train == 1).sum()} leak)")
    print(f"  Test set: {len(X_test)} valves ({(y_test == 1).sum()} leak)")

# ============================================================================
# 5. FEATURE SCALING
# ============================================================================

print("\n[5/7] Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("  StandardScaler fitted on training data")

# ============================================================================
# 6. TRAIN MODEL
# ============================================================================

print("\n[6/7] Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',  # Handle class imbalance
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

print("  Model trained successfully")

# ============================================================================
# 7. EVALUATE MODEL
# ============================================================================

print("\n[7/7] Evaluating model performance...")

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Training metrics
train_acc = accuracy_score(y_train, y_train_pred)
print(f"\n  TRAINING SET PERFORMANCE:")
print(f"    Accuracy: {train_acc:.1%}")

# Test metrics
if len(X_test) > 0:
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

    print(f"\n  TEST SET PERFORMANCE (C402 Cylinder 3 Holdout):")
    print(f"    Accuracy:  {test_acc:.1%}")
    print(f"    Precision: {test_precision:.1%}")
    print(f"    Recall:    {test_recall:.1%} (Critical: Don't miss leaks!)")
    print(f"    F1-Score:  {test_f1:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Leak")
    print(f"    Normal  [{cm[0,0]:>6} {cm[0,1]:>6}]")
    print(f"    Leak    [{cm[1,0]:>6} {cm[1,1]:>6}]")

    if (y_test == 1).any():
        leak_recall = test_recall
        if leak_recall == 1.0:
            print(f"\n  [OK] EXCELLENT: 100% of leaks detected in test set!")
        elif leak_recall >= 0.9:
            print(f"\n  [OK] GOOD: {leak_recall:.0%} of leaks detected in test set")
        else:
            print(f"\n  [WARNING] Only {leak_recall:.0%} of leaks detected")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  FEATURE IMPORTANCE:")
for idx, row in feature_importance.iterrows():
    print(f"    {row['feature']:25} {row['importance']:.3f}")

# ============================================================================
# 8. SAVE MODEL ARTIFACTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING MODEL ARTIFACTS")
print("=" * 80)

# Create output directory
model_dir = Path(__file__).parent.parent / 'data' / 'models'
model_dir.mkdir(parents=True, exist_ok=True)

# Save model
model_path = model_dir / 'leak_detector_model.pkl'
joblib.dump(model, model_path)
print(f"\n[OK] Model saved: {model_path}")

# Save scaler
scaler_path = model_dir / 'feature_scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"[OK] Scaler saved: {scaler_path}")

# Save metadata
metadata = {
    'model_type': 'RandomForestClassifier',
    'n_estimators': 100,
    'max_depth': 10,
    'features': feature_columns,
    'class_labels': ['Normal', 'Leak'],
    'training_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'unique_leak_valves': int(unique_leak_valves),
    'unique_normal_valves': int(unique_normal_valves),
    'test_accuracy': float(test_acc) if len(X_test) > 0 else None,
    'test_recall': float(test_recall) if len(X_test) > 0 else None,
    'feature_importance': {
        feat: float(imp) for feat, imp in zip(feature_columns, model.feature_importances_)
    }
}

metadata_path = model_dir / 'model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"[OK] Metadata saved: {metadata_path}")

# Copy to demo folder for deployment
demo_dir = Path(__file__).parent.parent / 'demo'
if demo_dir.exists():
    import shutil
    shutil.copy(model_path, demo_dir / 'leak_detector_model.pkl')
    shutil.copy(scaler_path, demo_dir / 'feature_scaler.pkl')
    print(f"\n[OK] Model artifacts copied to demo/ folder for Streamlit deployment")

print("\n" + "=" * 80)
print("RETRAINING COMPLETE!")
print("=" * 80)
print(f"\nSummary:")
print(f"  - Training data: {len(X_train)} valves ({unique_leak_valves} leak valves)")
print(f"  - Improvement: {unique_leak_valves - 7} more leak valves than original model")
if len(X_test) > 0:
    print(f"  - Test accuracy: {test_acc:.1%}")
    print(f"  - Leak recall: {test_recall:.1%}")
print(f"  - Model ready for Monday demo!")
