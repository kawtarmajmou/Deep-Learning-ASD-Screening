"""
MRI-based ASD Classification using ABIDE Dataset
=================================================
Uses functional brain network features (lambda2_star) from ABIDE fMRI data
to classify Autism Spectrum Disorder vs Normal controls.

Features used:
  - lambda2_star     : algebraic connectivity (main brain network metric)
  - thr_l2_maxcurv_s : optimal connectivity threshold
  - thr_knee_l2      : knee-point threshold
  - knee_score       : curvature score at knee
  - disconnected     : network fragmentation flag
  - n_roi_dropped    : number of dropped brain regions
  - AGE_AT_SCAN      : patient age
  - SEX              : patient sex
  - FIQ              : full-scale IQ

Run:
  python src/mri_model.py
"""

import os, sys, csv, json, pickle, warnings
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

ROOT        = Path(__file__).parent.parent
MRI_DIR     = ROOT / 'mri' / 'Sp'
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

DATASET_PATH    = MRI_DIR / 'dataset.csv'
PHENOTYPIC_PATH = MRI_DIR / 'Phenotypic_V1_0b_preprocessed1.csv'
MODEL_OUT       = RESULTS_DIR / 'mri_model.pkl'
RESULTS_OUT     = RESULTS_DIR / 'mri_results.json'

print("=" * 60)
print("  MRI-based ASD Classification (ABIDE Dataset)")
print("=" * 60)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_csv(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

print("\n[1] Loading data...")
brain_rows = load_csv(DATASET_PATH)
pheno_rows = load_csv(PHENOTYPIC_PATH)
print(f"    Brain network data : {len(brain_rows)} subjects")
print(f"    Phenotypic data    : {len(pheno_rows)} subjects")

# Build phenotypic lookup: SUB_ID -> dict
pheno_map = {}
for row in pheno_rows:
    try:
        sid = str(int(row['SUB_ID'])).zfill(7)
        pheno_map[sid] = row
    except:
        pass

print(f"    Phenotypic lookup  : {len(pheno_map)} entries")


# ── MERGE & BUILD FEATURE MATRIX ──────────────────────────────────────────────

print("\n[2] Merging datasets...")

def safe_float(val, default=np.nan):
    try:
        return float(val)
    except:
        return default

X_rows, y, matched_ids = [], [], []

for row in brain_rows:
    sid = str(row['id7']).zfill(7)
    if sid not in pheno_map:
        continue

    ph = pheno_map[sid]

    # Label: DX_GROUP 1=Autism, 2=Control -> 1=Autism, 0=Normal
    try:
        dx = int(ph['DX_GROUP'])
        label = 1 if dx == 1 else 0
    except:
        continue

    # Brain network features
    lambda2_star     = safe_float(row['lambda2_star'])
    thr_maxcurv      = safe_float(row['thr_l2_maxcurv_s'])
    thr_knee         = safe_float(row['thr_knee_l2'])
    knee_score       = safe_float(row['knee_score'])
    disconnected     = safe_float(row['disconnected'])
    n_roi_dropped    = safe_float(row['n_roi_dropped'])

    # Clinical/demographic features
    age  = safe_float(ph.get('AGE_AT_SCAN', ''))
    sex  = safe_float(ph.get('SEX', ''))
    fiq  = safe_float(ph.get('FIQ', ''))

    features = [
        lambda2_star, thr_maxcurv, thr_knee,
        knee_score, disconnected, n_roi_dropped,
        age, sex, fiq
    ]

    X_rows.append(features)
    y.append(label)
    matched_ids.append(sid)

X = np.array(X_rows, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print(f"    Matched subjects   : {len(y)}")
print(f"    Autism             : {np.sum(y == 1)}")
print(f"    Normal             : {np.sum(y == 0)}")
print(f"    Features per subject: {X.shape[1]}")

FEATURE_NAMES = [
    'lambda2_star', 'thr_l2_maxcurv_s', 'thr_knee_l2',
    'knee_score', 'disconnected', 'n_roi_dropped',
    'age', 'sex', 'fiq'
]


# ── CROSS-VALIDATION ──────────────────────────────────────────────────────────

print("\n[3] Running 5-Fold Cross-Validation...")

models = {
    'SVM (RBF)': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('clf',     SVC(kernel='rbf', C=10, gamma='scale',
                        probability=True, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf',     RandomForestClassifier(n_estimators=200, random_state=42,
                                           class_weight='balanced'))
    ]),
    'Gradient Boosting': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('clf',     GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, pipe in models.items():
    auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc',   n_jobs=1)
    f1_scores  = cross_val_score(pipe, X, y, cv=cv, scoring='f1',        n_jobs=1)
    acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy',  n_jobs=1)

    cv_results[name] = {
        'auc': round(float(np.mean(auc_scores)) * 100, 2),
        'f1' : round(float(np.mean(f1_scores))  * 100, 2),
        'acc': round(float(np.mean(acc_scores)) * 100, 2),
    }
    print(f"    {name:20s}  AUC={cv_results[name]['auc']:.1f}%  "
          f"F1={cv_results[name]['f1']:.1f}%  Acc={cv_results[name]['acc']:.1f}%")


# ── TRAIN BEST MODEL ON ALL DATA ──────────────────────────────────────────────

# Pick best model by AUC
best_name = max(cv_results, key=lambda k: cv_results[k]['auc'])
print(f"\n[4] Best model: {best_name}")

best_pipe = models[best_name]
best_pipe.fit(X, y)

# Save model + feature names
with open(MODEL_OUT, 'wb') as f:
    pickle.dump({'model': best_pipe, 'features': FEATURE_NAMES}, f)
print(f"    Model saved to {MODEL_OUT}")


# ── FEATURE IMPORTANCE (Random Forest) ───────────────────────────────────────

if 'Random Forest' in best_name:
    importances = best_pipe.named_steps['clf'].feature_importances_
    print("\n    Feature Importances:")
    for fname, imp in sorted(zip(FEATURE_NAMES, importances),
                              key=lambda x: -x[1]):
        bar = '#' * int(imp * 40)
        print(f"      {fname:22s} {imp:.3f}  {bar}")


# ── SAVE RESULTS ──────────────────────────────────────────────────────────────

results = {
    'dataset'       : 'ABIDE (fMRI functional connectivity)',
    'total_subjects': int(len(y)),
    'autism'        : int(np.sum(y == 1)),
    'normal'        : int(np.sum(y == 0)),
    'best_model'    : best_name,
    'cv_results'    : cv_results,
    'best_auc'      : cv_results[best_name]['auc'],
    'best_f1'       : cv_results[best_name]['f1'],
    'best_acc'      : cv_results[best_name]['acc'],
}

with open(RESULTS_OUT, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n    Results saved to {RESULTS_OUT}")

print("\n" + "=" * 60)
print("  FINAL MRI RESULTS")
print("=" * 60)
print(f"  Dataset  : ABIDE fMRI ({len(y)} subjects)")
print(f"  Best model: {best_name}")
print(f"  AUC      : {cv_results[best_name]['auc']}%")
print(f"  F1       : {cv_results[best_name]['f1']}%")
print(f"  Accuracy : {cv_results[best_name]['acc']}%")
print("=" * 60)
print("\nDone! You can now use this model in the website.")
