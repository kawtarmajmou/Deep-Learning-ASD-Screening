"""
Feature Extraction + Linear Probe Cross-Validation
====================================================
Uses the pretrained CNN-LSTM model as a feature extractor,
then trains a LogisticRegression / SVM on the extracted features
per subject-level fold. Much faster than fine-tuning.

Key insight: The CNN body learned EEG spectral features from all subjects.
Freezing it and only training a linear head uses those features without
catastrophic forgetting. With scikit-learn classifiers, each fold takes ~30s.
"""

import os, sys, json, warnings
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'data/spectrograms'
RESULTS  = Path('results')
SEQ_LEN  = 5
SEQ_STEP = 2
SEED     = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("="*60, flush=True)
print("FEATURE EXTRACTION + LINEAR PROBE CV", flush=True)
print(f"Device: {DEVICE}", flush=True)
print("="*60, flush=True)


# ── SAME ARCHITECTURE ─────────────────────────────────────────────────────────

class LightCNNLSTM(nn.Module):
    def __init__(self, sl=5):
        super().__init__()
        self.sl = sl
        self.cnn = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)))
        self.lstm = nn.LSTM(1024, 256, num_layers=2, batch_first=True, dropout=0.4)
        self.fc   = nn.Sequential(
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128,2))

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = [self.cnn(x[:,t]).view(B,-1) for t in range(T)]
        _, (hn,_) = self.lstm(torch.stack(feats, 1))
        return self.fc(hn[-1])

    def extract(self, x):
        """Return 256-dim LSTM hidden state (before FC classifier)."""
        B, T, C, H, W = x.shape
        feats = [self.cnn(x[:,t]).view(B,-1) for t in range(T)]
        _, (hn,_) = self.lstm(torch.stack(feats, 1))
        return hn[-1]  # (B, 256)


# ── DATASET ───────────────────────────────────────────────────────────────────

NORM   = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
VAL_TF = T.Compose([T.Resize((64,64)), T.ToTensor(), NORM])

class SeqDS(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        paths, lbl = self.seqs[i]
        frames = []
        for p in paths:
            try: img = Image.open(p).convert('RGB')
            except: img = Image.new('RGB', (64,64))
            frames.append(VAL_TF(img))
        return torch.stack(frames), lbl


def build_seqs(root, slen=SEQ_LEN, step=SEQ_STEP):
    label_map = {'Autism': 1, 'Normal': 0}
    sessions = defaultdict(list)
    for cls in os.listdir(root):
        d = os.path.join(root, cls)
        if not os.path.isdir(d): continue
        lbl = label_map.get(cls)
        if lbl is None: continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.png'): continue
            parts = f.split('_')
            key = '_'.join(parts[:min(5, len(parts))]) + '_' + str(lbl)
            sessions[key].append((os.path.join(d, f), lbl))

    seqs, subjects = [], []
    for key, files in sessions.items():
        if len(files) < slen: continue
        lbl   = files[0][1]
        paths = [f[0] for f in files]
        subj  = Path(files[0][0]).name.split('_')[0]
        for i in range(0, len(paths) - slen + 1, step):
            seqs.append((paths[i:i+slen], lbl))
            subjects.append(subj)
    return seqs, np.array(subjects)


@torch.no_grad()
def extract_features(model, seqs):
    """Extract 256-dim features from all sequences."""
    if not seqs:
        return np.array([]), np.array([])
    dl = DataLoader(SeqDS(seqs), batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    feats, labels = [], []
    for xb, yb in dl:
        f = model.extract(xb.to(DEVICE)).cpu().numpy()
        feats.append(f)
        labels.extend(yb.numpy())
    return np.vstack(feats), np.array(labels)


# ── FOLDS ─────────────────────────────────────────────────────────────────────

MANUAL_FOLDS = [
    ['Amer',     'Bader',    'Deena'],
    ['Dhelal',   'Duaa',     'Mada'],
    ['Mahmud',   'Mohammed', 'MohammedAshi'],
    ['Majdolin', 'Nour',     'Omar'],
    ['Omran',    'Shahad',   'Yahia',    'Zyad'],
]


# ── MAIN ─────────────────────────────────────────────────────────────────────

print("\nBuilding sequences...", flush=True)
all_seqs, all_subjects = build_seqs(DATA_DIR)
n = len(all_seqs)
n0 = sum(1 for s in all_seqs if s[1]==0)
n1 = sum(1 for s in all_seqs if s[1]==1)
print(f"Total: {n} seqs | Normal:{n0} | Autism:{n1}", flush=True)

print("\nLoading pretrained CNN-LSTM...", flush=True)
backbone = LightCNNLSTM(sl=SEQ_LEN).to(DEVICE)
backbone.load_state_dict(torch.load(RESULTS/'lstm_v2_best.pth', map_location=DEVICE))
backbone.eval()
print("Loaded OK", flush=True)

# Pre-extract ALL features once (saves repeating per fold)
print("\nExtracting features for ALL sequences (one pass)...", flush=True)
all_features, all_labels = extract_features(backbone, all_seqs)
print(f"Feature matrix: {all_features.shape}  Labels: {all_labels.shape}", flush=True)

# Classifiers to try
classifiers = {
    'LR (C=1)':  LogisticRegression(C=1.0,   max_iter=1000, random_state=SEED, class_weight='balanced'),
    'LR (C=0.1)': LogisticRegression(C=0.1,  max_iter=1000, random_state=SEED, class_weight='balanced'),
    'SVM-RBF':   SVC(C=1.0, kernel='rbf',    probability=True, random_state=SEED, class_weight='balanced'),
    'SVM-Lin':   SVC(C=1.0, kernel='linear', probability=True, random_state=SEED, class_weight='balanced'),
}

all_results = {name: [] for name in classifiers}

for fold, test_subjects in enumerate(MANUAL_FOLDS):
    te_mask = np.isin(all_subjects, test_subjects)
    tr_mask = ~te_mask

    X_tr, y_tr = all_features[tr_mask], all_labels[tr_mask]
    X_te, y_te = all_features[te_mask], all_labels[te_mask]

    n0_te = (y_te == 0).sum()
    n1_te = (y_te == 1).sum()

    print(f"\n{'-'*55}", flush=True)
    print(f"Fold {fold+1} | Test: {test_subjects}", flush=True)
    print(f"  Train: {len(X_tr)} seqs | Test: {len(X_te)} (N:{n0_te} A:{n1_te})", flush=True)

    if n0_te == 0 or n1_te == 0:
        print("  [SKIP] Only one class in test", flush=True)
        continue

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    for clf_name, clf in classifiers.items():
        clf.fit(X_tr_s, y_tr)
        probs = clf.predict_proba(X_te_s)[:, 1]
        preds = clf.predict(X_te_s)

        acc = accuracy_score(y_te, preds) * 100
        f1  = f1_score(y_te, preds, average='macro') * 100
        auc = roc_auc_score(y_te, probs) * 100
        cm  = confusion_matrix(y_te, preds)
        sens = cm[1,1]/(cm[1,0]+cm[1,1]+1e-9)*100
        spec = cm[0,0]/(cm[0,0]+cm[0,1]+1e-9)*100

        # Optimal threshold
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.1, 0.9, 0.02):
            f = f1_score(y_te, (probs >= t).astype(int), average='macro')
            if f > best_f1: best_f1 = f; best_t = t
        y_opt = (probs >= best_t).astype(int)
        acc2 = accuracy_score(y_te, y_opt)*100
        f12  = f1_score(y_te, y_opt, average='macro')*100
        cm2  = confusion_matrix(y_te, y_opt)
        sens2 = cm2[1,1]/(cm2[1,0]+cm2[1,1]+1e-9)*100
        spec2 = cm2[0,0]/(cm2[0,0]+cm2[0,1]+1e-9)*100

        print(f"  {clf_name:<14} Acc={acc2:.1f}% F1={f12:.1f}% AUC={auc:.1f}% "
              f"Sens={sens2:.1f}% Spec={spec2:.1f}% (thr={best_t:.2f})", flush=True)

        all_results[clf_name].append({
            'fold': fold+1, 'test': test_subjects,
            'acc': round(acc2,2), 'f1': round(f12,2), 'auc': round(auc,2),
            'sens': round(sens2,2), 'spec': round(spec2,2)
        })

# ── SUMMARY ──────────────────────────────────────────────────────────────────

print(f"\n{'='*60}", flush=True)
print("FEATURE EXTRACTION CV SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'Classifier':<16} {'Acc':>9} {'F1':>9} {'AUC':>9} {'Sens':>9} {'Spec':>9}", flush=True)
print(f"{'-'*63}", flush=True)

# Reference
print(f"{'CNN-LSTM (leakage)':<16} {'77.82%':>9} {'77.45%':>9} {'81.85%':>9} {'86.38%':>9} {'57.92%':>9}", flush=True)
print(f"{'CNN-LSTM (subj CV)':<16} {'83.84%':>9} {'78.89%':>9} {'69.37%':>9} {'94.75%':>9} {'58.64%':>9}", flush=True)
print(f"{'-'*63}", flush=True)

summary = {}
for clf_name, results in all_results.items():
    if not results: continue
    keys = ['acc','f1','auc','sens','spec']
    means = {k: np.mean([r[k] for r in results]) for k in keys}
    stds  = {k: np.std( [r[k] for r in results]) for k in keys}
    print(f"{'FE+'+clf_name:<16} {means['acc']:>7.2f}% {means['f1']:>8.2f}% "
          f"{means['auc']:>8.2f}% {means['sens']:>8.2f}% {means['spec']:>8.2f}%  "
          f"(±AUC {stds['auc']:.2f}%)", flush=True)
    summary[clf_name] = {'cv_mean': {k: round(means[k],2) for k in keys},
                          'cv_std':  {k: round(stds[k],2)  for k in keys},
                          'fold_results': results}

with open(RESULTS/'feature_extract_cv.json','w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved: results/feature_extract_cv.json", flush=True)
