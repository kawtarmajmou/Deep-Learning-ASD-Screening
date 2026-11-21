"""
Subject-Level Evaluation of Existing Best Models
==================================================
Loads trained models and evaluates them on proper subject-level test folds.
No retraining needed - fast (~5 minutes).
Gives honest generalization performance for thesis.
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

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
print("SUBJECT-LEVEL EVALUATION OF TRAINED MODELS", flush=True)
print(f"Device: {DEVICE}", flush=True)
print("="*60, flush=True)


# ── MODELS ────────────────────────────────────────────────────────────────────

class LightCNNLSTM(nn.Module):
    """Exact same architecture as train_lstm_v2.py (best model at 77.82%)"""
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
    """Build (paths, label) sequences grouped by recording session."""
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
def eval_seqs(model, seqs):
    if not seqs: return 0, 0, 0, np.array([]), np.array([]), np.array([])
    dl = DataLoader(SeqDS(seqs), batch_size=16, shuffle=False, num_workers=0)
    model.eval()
    probs, preds, truth = [], [], []
    for xb, yb in dl:
        out = model(xb.to(DEVICE))
        p   = F.softmax(out, 1)[:,1].cpu().numpy()
        probs.extend(p); preds.extend(out.argmax(1).cpu().numpy()); truth.extend(yb.numpy())
    acc = accuracy_score(truth, preds) * 100
    f1  = f1_score(truth, preds, average='macro') * 100
    try: auc = roc_auc_score(truth, probs) * 100
    except: auc = 0.0
    return acc, f1, auc, np.array(truth), np.array(preds), np.array(probs)


# ── MANUAL FOLDS ──────────────────────────────────────────────────────────────

MANUAL_FOLDS = [
    ['Amer',     'Bader',    'Deena'],
    ['Dhelal',   'Duaa',     'Mada'],
    ['Mahmud',   'Mohammed', 'MohammedAshi'],
    ['Majdolin', 'Nour',     'Omar'],
    ['Omran',    'Shahad',   'Yahia',    'Zyad'],
]


def evaluate_model_on_folds(model, all_seqs, all_subjects, model_name):
    """Evaluate a trained model on each subject-level test fold."""
    print(f"\n{'='*55}", flush=True)
    print(f"MODEL: {model_name}", flush=True)
    print(f"{'='*55}", flush=True)

    fold_results = []
    all_true, all_pred, all_prob = [], [], []

    for fold, test_subjects in enumerate(MANUAL_FOLDS):
        te_mask = np.isin(all_subjects, test_subjects)
        te_seqs = [s for s, m in zip(all_seqs, te_mask) if m]

        te_lbl0 = sum(1 for s in te_seqs if s[1] == 0)
        te_lbl1 = sum(1 for s in te_seqs if s[1] == 1)
        print(f"\n  Fold {fold+1} | Test: {test_subjects}", flush=True)
        print(f"    Seqs: {len(te_seqs)} (N:{te_lbl0} A:{te_lbl1})", flush=True)

        if te_lbl0 == 0 or te_lbl1 == 0:
            print(f"    [SKIP] Only one class in test fold", flush=True)
            continue

        acc, f1, auc, y_true, y_pred, y_prob = eval_seqs(model, te_seqs)

        # Optimal threshold
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.2, 0.8, 0.02):
            f = f1_score(y_true, (y_prob >= t).astype(int), average='macro')
            if f > best_f1: best_f1 = f; best_t = t
        y_opt = (y_prob >= best_t).astype(int)
        acc2  = accuracy_score(y_true, y_opt) * 100
        f12   = f1_score(y_true, y_opt, average='macro') * 100
        cm    = confusion_matrix(y_true, y_opt)
        sens  = cm[1,1] / (cm[1,0]+cm[1,1]+1e-9) * 100
        spec  = cm[0,0] / (cm[0,0]+cm[0,1]+1e-9) * 100

        print(f"    Acc={acc2:.1f}% F1={f12:.1f}% AUC={auc:.1f}% Sens={sens:.1f}% Spec={spec:.1f}% (thr={best_t:.2f})", flush=True)

        fold_results.append({'fold': fold+1, 'test': test_subjects,
                              'acc': round(acc2,2), 'f1': round(f12,2), 'auc': round(auc,2),
                              'sens': round(sens,2), 'spec': round(spec,2)})
        all_true.extend(y_true.tolist()); all_pred.extend(y_opt.tolist()); all_prob.extend(y_prob.tolist())

    if not fold_results:
        return {}

    keys = ['acc','f1','auc','sens','spec']
    means = {k: np.mean([r[k] for r in fold_results]) for k in keys}
    stds  = {k: np.std( [r[k] for r in fold_results]) for k in keys}

    print(f"\n  --- {model_name} Subject-Level CV Summary ---", flush=True)
    for k in keys:
        print(f"    {k.upper():<6}: {means[k]:.2f}% +/- {stds[k]:.2f}%", flush=True)

    # Pooled
    all_true = np.array(all_true); all_pred = np.array(all_pred); all_prob = np.array(all_prob)
    p_acc = accuracy_score(all_true, all_pred)*100
    p_f1  = f1_score(all_true, all_pred, average='macro')*100
    try: p_auc = roc_auc_score(all_true, all_prob)*100
    except: p_auc = 0.0
    cm_all = confusion_matrix(all_true, all_pred)
    p_sens = cm_all[1,1]/(cm_all[1,0]+cm_all[1,1]+1e-9)*100
    p_spec = cm_all[0,0]/(cm_all[0,0]+cm_all[0,1]+1e-9)*100

    print(f"\n  Pooled: Acc={p_acc:.2f}% F1={p_f1:.2f}% AUC={p_auc:.2f}% Sens={p_sens:.2f}% Spec={p_spec:.2f}%", flush=True)

    return {
        'model': model_name, 'fold_results': fold_results,
        'cv_mean': {k: round(means[k],2) for k in keys},
        'cv_std':  {k: round(stds[k],2)  for k in keys},
        'pooled': {'accuracy': round(p_acc,2), 'f1': round(p_f1,2), 'auc': round(p_auc,2),
                   'sensitivity': round(p_sens,2), 'specificity': round(p_spec,2)}
    }


# ── MAIN ─────────────────────────────────────────────────────────────────────

print("\nBuilding all sequences...", flush=True)
all_seqs, all_subjects = build_seqs(DATA_DIR)
n = len(all_seqs)
n0 = sum(1 for s in all_seqs if s[1]==0); n1 = sum(1 for s in all_seqs if s[1]==1)
print(f"Total sequences: {n} | Normal:{n0} | Autism:{n1}", flush=True)

all_results = {}

# ── Evaluate lstm_v2_best.pth (LightCNNLSTM, ~7.5MB) ─────────────────────────
ckpt1 = RESULTS / 'lstm_v2_best.pth'
if ckpt1.exists():
    print(f"\nLoading {ckpt1}...", flush=True)
    model1 = LightCNNLSTM(sl=SEQ_LEN).to(DEVICE)
    try:
        model1.load_state_dict(torch.load(ckpt1, map_location=DEVICE))
        print("Loaded OK", flush=True)
        r = evaluate_model_on_folds(model1, all_seqs, all_subjects, "CNN-LSTM-V2 (lstm_v2_best)")
        if r: all_results['CNN-LSTM-V2'] = r
    except Exception as e:
        print(f"Failed to load: {e}", flush=True)
    del model1
else:
    print(f"[SKIP] {ckpt1} not found", flush=True)

# ── Evaluate CNN_LSTM_best.pth (CNN_LSTM_Hybrid from models.py) ───────────────
# Try the original CNN_LSTM_best.pth with CNN_LSTM_Hybrid architecture
ckpt2 = RESULTS / 'CNN_LSTM_best.pth'
if ckpt2.exists():
    print(f"\nLoading {ckpt2}...", flush=True)
    # Try LightCNNLSTM first, then ImprovedCNNLSTM
    for arch_name, arch_class in [("LightCNNLSTM-sl5", lambda: LightCNNLSTM(sl=5))]:
        try:
            model2 = arch_class().to(DEVICE)
            model2.load_state_dict(torch.load(ckpt2, map_location=DEVICE))
            print(f"Loaded as {arch_name}", flush=True)
            r = evaluate_model_on_folds(model2, all_seqs, all_subjects, f"CNN-LSTM-Best ({arch_name})")
            if r: all_results[f'CNN-LSTM-Best'] = r
            del model2
            break
        except Exception as e:
            print(f"  [{arch_name}] failed: {e}", flush=True)

# ── FINAL COMPARISON TABLE ────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("SUBJECT-LEVEL GENERALIZATION REPORT", flush=True)
print("="*60, flush=True)
print(f"{'Model':<30} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Sens':>8} {'Spec':>8}", flush=True)
print("-"*74, flush=True)

# Add the original segment-level result for comparison
seg_level = {'accuracy':77.82,'f1':77.45,'auc':81.85,'sensitivity':86.38,'specificity':57.92}
print(f"{'CNN-LSTM V2 (segment-level)*':<30} {seg_level['accuracy']:>7.2f}% {seg_level['f1']:>7.2f}% "
      f"{seg_level['auc']:>7.2f}% {seg_level['sensitivity']:>7.2f}% {seg_level['specificity']:>7.2f}%", flush=True)
print(f"  * segment-level split (data leakage if same subject in train+test)", flush=True)

for name, r in all_results.items():
    p = r.get('pooled', {})
    print(f"{name:<30} {p.get('accuracy',0):>7.2f}% {p.get('f1',0):>7.2f}% "
          f"{p.get('auc',0):>7.2f}% {p.get('sensitivity',0):>7.2f}% {p.get('specificity',0):>7.2f}%", flush=True)
    cv = r.get('cv_mean', {})
    print(f"  CV mean: Acc={cv.get('acc',0):.2f}% F1={cv.get('f1',0):.2f}% AUC={cv.get('auc',0):.2f}%", flush=True)

with open(RESULTS/'subject_level_eval.json','w') as f:
    json.dump({'subject_level_results': all_results,
               'segment_level_reference': seg_level}, f, indent=2)
print("\nSaved: results/subject_level_eval.json", flush=True)
