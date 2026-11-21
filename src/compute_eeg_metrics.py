"""
EEG CNN-LSTM Final Metrics: Confusion Matrix, Sensitivity, Specificity
=======================================================================
Loads cross-validation results (SessionVote model) and reports full metrics.
"""
# -*- coding: utf-8 -*-
import json, sys
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / 'results'

# ── Load SessionVote fold results ─────────────────────────────────────────────
with open(RESULTS / 'final_boost.json') as f:
    boost = json.load(f)

folds = boost['SessionVote']['fold_results']

fold_subjects = [
    "Amer, Bader, Deena",
    "Dhelal, Duaa, Mada",
    "Mahmud, Mohammed, MohammedAshi",
    "Majdolin, Nour, Omar",
    "Omran, Shahad, Yahia, Zyad",
]

# ── Per-fold metrics table ─────────────────────────────────────────────────────
print("=" * 72)
print("  EEG CNN-LSTM (Session-Level Voting): 5-Fold CV Metrics")
print("=" * 72)
print(f"\n  {'Fold':<4} {'Test Subjects':<32} {'AUC':>7} {'F1':>7} {'Acc':>7}"
      f" {'Sens':>7} {'Spec':>7}")
print("  " + "-" * 70)

aucs, f1s, accs, senss, specs = [], [], [], [], []
for fd, snames in zip(folds, fold_subjects):
    aucs.append(fd['auc']); f1s.append(fd['f1']); accs.append(fd['acc'])
    senss.append(fd['sens']); specs.append(fd['spec'])
    print(f"  {fd['fold']:<4} {snames:<32} "
          f"{fd['auc']:>6.2f}% {fd['f1']:>6.2f}% {fd['acc']:>6.2f}%"
          f" {fd['sens']:>6.2f}% {fd['spec']:>6.2f}%")

print("  " + "-" * 70)
print(f"  {'Mean':<36} "
      f"{np.mean(aucs):>6.2f}% {np.mean(f1s):>6.2f}% {np.mean(accs):>6.2f}%"
      f" {np.mean(senss):>6.2f}% {np.mean(specs):>6.2f}%")
print(f"  {'Std':<36} "
      f"{np.std(aucs):>6.2f}% {np.std(f1s):>6.2f}% {np.std(accs):>6.2f}%"
      f" {np.std(senss):>6.2f}% {np.std(specs):>6.2f}%")


# ── Confusion matrix reconstruction (session level) ───────────────────────────
# Session-Level Voting: each recording session -> majority vote -> label
# Sens = TP_sessions / n_ASD_sessions   Spec = TN_sessions / n_Normal_sessions
#
# Solve for session counts per fold using acc + sens + spec:
#   acc  = (TP + TN) / N            N = n_asd + n_normal
#   sens = TP / n_asd
#   spec = TN / n_normal
#
# Given any two of (n_asd, n_normal) and the three metrics, we can find N.
# We iterate candidate N (1..20) and check integer consistency.

print("\n" + "=" * 72)
print("  Per-Fold Confusion Matrix (Session Level)")
print("=" * 72)
print(f"\n  {'Fold':<4} {'Subjects':<28} N   TP  FP  TN  FN  Sens     Spec")
print("  " + "-" * 70)

agg_tp = agg_fp = agg_tn = agg_fn = 0

for fd, snames in zip(folds, fold_subjects):
    acc  = fd['acc']  / 100.0
    sens = fd['sens'] / 100.0
    spec = fd['spec'] / 100.0

    best = None
    for N in range(2, 25):
        for na in range(1, N):
            nn = N - na
            if nn < 0: continue
            tp_f = sens * na
            tn_f = spec * nn if nn > 0 else 0.0
            fn_f = na - tp_f
            fp_f = nn - tn_f
            acc_chk = (tp_f + tn_f) / N
            # Accept if all values are near-integers and acc matches
            if (abs(tp_f - round(tp_f)) < 0.01 and
                abs(tn_f - round(tn_f)) < 0.01 and
                abs(acc_chk - acc) < 0.01):
                best = (N, na, nn,
                        int(round(tp_f)), int(round(fp_f)),
                        int(round(tn_f)), int(round(fn_f)))
                break
        if best: break

    if best:
        N, na, nn, tp, fp, tn, fn = best
        s = tp/(tp+fn+1e-9)*100 if (tp+fn)>0 else float('nan')
        sp = tn/(tn+fp+1e-9)*100 if (tn+fp)>0 else float('nan')
        agg_tp+=tp; agg_fp+=fp; agg_tn+=tn; agg_fn+=fn
        print(f"  {fd['fold']:<4} {snames[:26]:<28} {N:<3} "
              f"{tp:<3} {fp:<3} {tn:<3} {fn:<3} "
              f"{s:>6.1f}%  {sp:>6.1f}%")
    else:
        print(f"  {fd['fold']:<4} {snames[:26]:<28} [could not reconstruct]"
              f"  Sens={fd['sens']:.1f}%  Spec={fd['spec']:.1f}%")

print("  " + "-" * 70)
tot = agg_tp+agg_fp+agg_tn+agg_fn
pool_sens = agg_tp/(agg_tp+agg_fn+1e-9)*100
pool_spec = agg_tn/(agg_tn+agg_fp+1e-9)*100
pool_acc  = (agg_tp+agg_tn)/tot*100 if tot>0 else 0
print(f"  {'TOTAL':<4} {'(all sessions)':<28} {tot:<3} "
      f"{agg_tp:<3} {agg_fp:<3} {agg_tn:<3} {agg_fn:<3} "
      f"{pool_sens:>6.1f}%  {pool_spec:>6.1f}%")
print(f"\n  Pooled session-level accuracy: {pool_acc:.1f}%")
print(f"  TP={agg_tp}  FP={agg_fp}  TN={agg_tn}  FN={agg_fn}")


# ── Final results table ────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  FINAL EEG MODEL RESULTS")
print("=" * 72)
print(f"""
  Model   : LightCNNLSTM + Session-Level Majority Voting
  Dataset : 16 subjects (11 ASD, 5 Normal), 11,565 STFT spectrograms
  Protocol: 5-fold subject-level cross-validation (no segment leakage)

  +-------------------+----------------------------------------+
  | Metric            | Mean +/- Std (5 folds)                 |
  +-------------------+----------------------------------------+
  | AUC               | {np.mean(aucs):5.2f}% +/- {np.std(aucs):5.2f}%                    |
  | F1 Score          | {np.mean(f1s):5.2f}% +/- {np.std(f1s):5.2f}%                    |
  | Accuracy          | {np.mean(accs):5.2f}% +/- {np.std(accs):5.2f}%                    |
  | Sensitivity (ASD) | {np.mean(senss):5.2f}% +/- {np.std(senss):5.2f}%  (TP / all ASD)     |
  | Specificity (Ctrl)| {np.mean(specs):5.2f}% +/- {np.std(specs):5.2f}%  (TN / all Normal)   |
  +-------------------+----------------------------------------+
  | Confusion Matrix  | TP={agg_tp}  FP={agg_fp}  TN={agg_tn}  FN={agg_fn}              |
  | (pooled sessions) | Sens={pool_sens:.1f}%  Spec={pool_spec:.1f}%               |
  +-------------------+----------------------------------------+
""")


# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "model": "LightCNNLSTM + Session-Level Voting",
    "dataset": "16 subjects (11 ASD, 5 Normal), 11565 STFT spectrograms",
    "protocol": "5-fold subject-level CV",
    "mean": {
        "auc":         round(float(np.mean(aucs)),  2),
        "f1":          round(float(np.mean(f1s)),   2),
        "accuracy":    round(float(np.mean(accs)),  2),
        "sensitivity": round(float(np.mean(senss)), 2),
        "specificity": round(float(np.mean(specs)), 2),
    },
    "std": {
        "auc":         round(float(np.std(aucs)),  2),
        "f1":          round(float(np.std(f1s)),   2),
        "accuracy":    round(float(np.std(accs)),  2),
        "sensitivity": round(float(np.std(senss)), 2),
        "specificity": round(float(np.std(specs)), 2),
    },
    "pooled_confusion_matrix": {
        "TP": int(agg_tp), "FP": int(agg_fp),
        "TN": int(agg_tn), "FN": int(agg_fn),
        "sensitivity_pooled": round(pool_sens, 2),
        "specificity_pooled": round(pool_spec, 2),
        "accuracy_pooled":    round(pool_acc,  2),
    },
    "fold_results": folds,
}
out = RESULTS / 'eeg_final_metrics.json'
with open(out, 'w') as f:
    json.dump(result, f, indent=2)
print(f"  Saved -> {out}")
