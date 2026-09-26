"""
06_recalibration.py  -  runs anywhere, seconds, no GPU, no retraining

Threshold recalibration on MR-ART with network weights frozen.
Replaces the old Table 3 (hardcoded F1 = 0.813) with a real experiment.

For each label budget N, draws N MR-ART subjects at random, picks the
F1-maximising threshold on their scans, and evaluates that threshold on
all remaining scans. Draws are by SUBJECT, so no subject contributes to both
threshold selection and evaluation.

Input : results/predictions/mrart_zeroshot_scan_probs.csv   (from script 05)
Output: results/table3_recalibration.csv
        results/recalibration_raw.csv
        results/figures/recalibration_curve.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------- CONFIG
# ---------------------------------------------------------------- PATHS
# Defaults are relative to the repository root, so the pipeline runs from a
# clone with no edits. Override any of them with an environment variable:
#   export MRIQC_ROOT=/path/to/project
#   export MRIQC_SLICES=/path/to/slices
#   export MRIQC_ABIDE_NIFTI=/path/to/ABIDE          (script 01 only)
ROOT   = os.environ.get("MRIQC_ROOT",   os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SLICES_ROOT = os.environ.get("MRIQC_SLICES", os.path.join(ROOT, "slices"))
COHORT  = os.path.join(ROOT, "cohort")
RESULTS = os.path.join(ROOT, "results")
MODELS  = os.path.join(ROOT, "models")
PROBS   = os.path.join(RESULTS, "predictions", "mrart_zeroshot_scan_probs.csv")
BUDGETS = [5, 10, 20, 40, 60]      # labeled target subjects
N_DRAWS = 500
SEED    = 42
# ------------------------------------------------------------------------

os.makedirs(f"{RESULTS}/figures", exist_ok=True)

d = pd.read_csv(PROBS)
y = d.y.values.astype(int)
s = d.p.values
g = d.subject_id.values
print(f"{len(d)} scans | {len(set(g))} subjects | motion {y.sum()} | "
      f"AUROC {roc_auc_score(y, s):.3f}")


def f1_curve(yy, ss):
    """F1 at every distinct score used as threshold (vectorised)."""
    o = np.argsort(-ss)
    ys, sv = yy[o], ss[o]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[sv[1:] != sv[:-1], True]
    tp, fp, thr = tp[last], fp[last], sv[last]
    prec, rec = tp / (tp + fp), tp / ys.sum()
    f1 = np.where(tp > 0, 2 * prec * rec / (prec + rec), 0)
    return thr, f1


def at(yy, ss, t):
    """F1, precision, recall at a fixed threshold."""
    pr = ss >= t
    tp = (pr & (yy == 1)).sum()
    fp = (pr & (yy == 0)).sum()
    fn = (~pr & (yy == 1)).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


rng = np.random.default_rng(SEED)
subs = np.unique(g)
rows = []
for N in BUDGETS:
    for draw in range(N_DRAWS):
        cal = rng.choice(subs, N, replace=False)
        m = np.isin(g, cal)
        if len(set(y[m])) < 2:          # need both classes to set a threshold
            continue
        thr, f1 = f1_curve(y[m], s[m])
        t = thr[np.argmax(f1)]
        f, p, r = at(y[~m], s[~m], t)
        rows.append({"N": N, "draw": draw, "threshold": t,
                     "F1": f, "Precision": p, "Recall": r,
                     "F1_at_0.5": at(y[~m], s[~m], 0.5)[0],
                     "F1_oracle": f1_curve(y[~m], s[~m])[1].max()})

raw = pd.DataFrame(rows)
raw.to_csv(f"{RESULTS}/recalibration_raw.csv", index=False)

summ = (raw.groupby("N")
        .agg(draws=("F1", "size"),
             threshold_median=("threshold", "median"),
             F1_mean=("F1", "mean"), F1_sd=("F1", "std"),
             F1_p10=("F1", lambda x: np.percentile(x, 10)),
             Precision=("Precision", "mean"), Recall=("Recall", "mean"),
             F1_at_05=("F1_at_0.5", "mean"), F1_oracle=("F1_oracle", "mean"))
        .reset_index())
summ["gap_closed_pct"] = (100 * (summ.F1_mean - summ.F1_at_05)
                          / (summ.F1_oracle - summ.F1_at_05))
prev = y.mean()
summ["F1_flag_all"] = 2 * prev / (1 + prev)

print(summ.round(3).to_string(index=False))
summ.to_csv(f"{RESULTS}/table3_recalibration.csv", index=False)

plt.figure(figsize=(7, 4.5))
plt.errorbar(summ.N, summ.F1_mean, yerr=summ.F1_sd, fmt="o-", capsize=3,
             label="recalibrated threshold (weights frozen)")
plt.axhline(summ.F1_oracle.mean(), ls=":", c="green", label="oracle threshold")
plt.axhline(summ.F1_at_05.mean(), ls="--", c="k", label="inherited 0.5")
plt.axhline(summ.F1_flag_all.iloc[0], ls="-.", c="gray", label="flag every scan")
plt.xlabel("labeled MR-ART subjects used to set threshold")
plt.ylabel("F1 on held-out scans")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{RESULTS}/figures/recalibration_curve.png", dpi=300)
print(f"\nwrote {RESULTS}/table3_recalibration.csv and figure")
