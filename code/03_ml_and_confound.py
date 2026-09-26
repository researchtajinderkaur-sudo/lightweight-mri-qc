"""
03_ml_and_confound.py  -  runs anywhere, seconds, no images needed

Trains the four IQM-based baselines on the rebuilt ABIDE cohort and quantifies
how much of their performance is scanning-site identity rather than image
quality.

Replaces the single 80/20 split (which left 6 positives in test) with repeated
stratified CV pooled out-of-fold, plus leave-one-site-out.

Writes results/table1_abide.csv, results/site_confound.csv,
results/leave_one_site_out.csv and results/predictions/abide_oof_probs.csv.
"""

import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, recall_score, precision_score,
                             balanced_accuracy_score)

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
COHORT_CSV = os.path.join(COHORT, "abide_cohort_extractable.csv")
N_SPLITS, N_REPEATS = 5, 10
SEED = 0
# ------------------------------------------------------------------------

IQM = ["anat_cnr", "anat_efc", "anat_fber", "anat_fwhm", "anat_qi1", "anat_snr"]
os.makedirs(os.path.join(RESULTS, "predictions"), exist_ok=True)

d = pd.read_csv(COHORT_CSV).dropna(subset=IQM).reset_index(drop=True)
X = d[IQM].values
y = (d.quality == 0).astype(int).values          # 1 = BAD (positive class)
site = d.SITE_ID.values
prev = y.mean()

print(f"cohort: n={len(d)}  BAD={y.sum()}  GOOD={(1-y).sum()}  "
      f"prevalence={prev:.4f}")
print(f"chance AUPRC = {prev:.4f}   (report AUPRC against this, not raw F1)\n")


def models():
    return {
        "RandomForest": RandomForestClassifier(300, class_weight="balanced",
                                               random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=5,
                                               class_weight="balanced",
                                               random_state=42),
        "SVM":          SVC(kernel="rbf", class_weight="balanced",
                            probability=True, random_state=42),
        "kNN":          KNeighborsClassifier(n_neighbors=5),
    }


def pooled_oof(Xin, name_filter=None):
    """Repeated stratified CV, averaged out-of-fold probabilities."""
    out = {}
    for name, _ in models().items():
        if name_filter and name != name_filter:
            continue
        acc = np.zeros(len(y)); cnt = np.zeros(len(y))
        cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                     random_state=SEED)
        for tr, te in cv.split(Xin, y):
            sc = StandardScaler().fit(Xin[tr])
            m = models()[name].fit(sc.transform(Xin[tr]), y[tr])
            acc[te] += m.predict_proba(sc.transform(Xin[te]))[:, 1]
            cnt[te] += 1
        out[name] = acc / cnt
    return out


def score(y_true, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, p),
        "AUPRC": average_precision_score(y_true, p),
        "BalAcc": balanced_accuracy_score(y_true, pred),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
    }


# ---- Table 1: IQM baselines -------------------------------------------
print("=== Table 1: IQM baselines, 5x10 repeated CV pooled out-of-fold ===")
oof = pooled_oof(X)
rows = []
for name, p in oof.items():
    r = {"model": name, **score(y, p)}
    r["AUPRC_lift"] = r["AUPRC"] / prev
    rows.append(r)
t1 = pd.DataFrame(rows).sort_values("AUPRC", ascending=False)
print(t1.round(3).to_string(index=False))
t1.to_csv(os.path.join(RESULTS, "table1_abide.csv"), index=False)

pd.DataFrame({"scan_id": d.scan_id, "SITE_ID": site, "y_bad": y,
              **{f"p_{k}": v for k, v in oof.items()}}
             ).to_csv(os.path.join(RESULTS, "predictions",
                                   "abide_oof_probs.csv"), index=False)

# ---- Site confound ----------------------------------------------------
print("\n=== Site confound: what does site alone explain? ===")
S = OneHotEncoder(sparse_output=False, handle_unknown="ignore"
                  ).fit_transform(d[["SITE_ID"]])


def rf_cv(Xin):
    acc = np.zeros(len(y))
    for tr, te in StratifiedKFold(N_SPLITS, shuffle=True,
                                  random_state=SEED).split(Xin, y):
        sc = StandardScaler().fit(Xin[tr])
        m = RandomForestClassifier(300, class_weight="balanced",
                                   random_state=1).fit(sc.transform(Xin[tr]), y[tr])
        acc[te] = m.predict_proba(sc.transform(Xin[te]))[:, 1]
    return acc


conf = []
for label, mat in [("SITE only", S), ("IQM only", X),
                   ("IQM + SITE", np.hstack([X, S]))]:
    p = rf_cv(mat)
    conf.append({"features": label, **score(y, p)})
tc = pd.DataFrame(conf)
print(tc.round(3).to_string(index=False))
tc.to_csv(os.path.join(RESULTS, "site_confound.csv"), index=False)

gain = tc.loc[1, "AUPRC"] - tc.loc[0, "AUPRC"]
print(f"\n  IQM gain over site-only: AUPRC +{gain:.3f}")
print("  -> report the SITE-only row as a control baseline in the paper.")
print("     If IQM barely exceeds it, the model is a site classifier.")

# ---- Leave-one-site-out ----------------------------------------------
print("\n=== Leave-one-site-out (Random Forest on IQMs) ===")
rows = []
for s_ in sorted(set(site)):
    tr, te = site != s_, site == s_
    n_bad = int(y[te].sum())
    if len(set(y[te])) < 2:
        rows.append({"site": s_, "n": int(te.sum()), "n_BAD": n_bad,
                     "AUROC": np.nan, "AUPRC": np.nan})
        continue
    sc = StandardScaler().fit(X[tr])
    m = RandomForestClassifier(300, class_weight="balanced",
                               random_state=42).fit(sc.transform(X[tr]), y[tr])
    p = m.predict_proba(sc.transform(X[te]))[:, 1]
    rows.append({"site": s_, "n": int(te.sum()), "n_BAD": n_bad,
                 "AUROC": roc_auc_score(y[te], p),
                 "AUPRC": average_precision_score(y[te], p)})
lo = pd.DataFrame(rows)
print(lo.round(3).to_string(index=False))
lo.to_csv(os.path.join(RESULTS, "leave_one_site_out.csv"), index=False)

v = lo.AUROC.dropna()
print(f"\n  evaluable sites: {len(v)}/{len(lo)}   "
      f"mean AUROC={v.mean():.3f} sd={v.std():.3f}")
print(f"\nwrote results to {RESULTS}")
