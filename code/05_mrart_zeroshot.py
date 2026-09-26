"""
05_mrart_zeroshot.py  -  RUN IN COLAB

Zero-shot external validation on MR-ART. This replaces Table 2, whose numbers
in the current manuscript were hardcoded literals.

Reports AUROC and AUPRC with subject-level bootstrap CIs, so transfer failure
can be separated into two distinct causes:
  * representation failure   -> AUROC near 0.5, signal did not transfer
  * operating-point failure  -> AUROC high, F1 low, threshold miscalibrated

The v4 model outputs P(BAD) directly (positive class = BAD), so there is no
1 - sigmoid flip. Grouping is by scan_id (subject + acquisition), never by
subject alone - collapsing a subject's standard and motion acquisitions into
one unit is what produced the 144-unit result in the old notebook.
"""

import os, re, glob, json
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support, roc_curve,
                             precision_recall_curve)
import matplotlib.pyplot as plt

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
MRART   = os.path.join(SLICES_ROOT, "MR_ART_SLICES")   # contains GOOD/ and BAD/
MODEL   = os.path.join(MODELS, "abide_cnn_v4.keras")
NORM    = os.path.join(MODELS, "abide_cnn_v4_norm.json")
N_BOOT  = 5000
RNG     = np.random.default_rng(42)
# ------------------------------------------------------------------------

os.makedirs(f"{RESULTS}/predictions", exist_ok=True)
os.makedirs(f"{RESULTS}/figures", exist_ok=True)

# Filename conventions, tried in order. The acquisition tag MUST be captured -
# without it a subject's clean and motion scans cannot be told apart.
PATTERNS = [
    r"^(?P<sub>sub-[A-Za-z0-9]+)_(?P<acq>acq-[A-Za-z0-9]+)_slice(?P<k>\d+)$",
    r"^(?P<sub>sub-[A-Za-z0-9]+)_(?P<acq>[A-Za-z0-9]+)_slice(?P<k>\d+)$",
    r"^(?P<sub>sub-[A-Za-z0-9]+)_slice(?P<k>\d+)$",          # no acq - fatal
]

rows, used = [], None
for cls, lab in (("GOOD", 0), ("BAD", 1)):
    for p in sorted(glob.glob(f"{MRART}/{cls}/*.png")):
        stem = os.path.splitext(os.path.basename(p))[0]
        for rx in PATTERNS:
            m = re.match(rx, stem)
            if m:
                used = used or rx
                g = m.groupdict()
                acq = g.get("acq", "NONE")
                rows.append({"path": p, "y": lab, "subject_id": g["sub"],
                             "acq": acq, "scan_id": f"{g['sub']}_{acq}"})
                break
        else:
            raise ValueError(f"unparsed filename: {stem}")

mr = pd.DataFrame(rows)
if len(mr) == 0:
    raise SystemExit(
        f"No PNGs found under {MRART}/GOOD and {MRART}/BAD.\n"
        f"  Check the MRART path in CONFIG matches your unzipped folder, e.g.\n"
        f"  set MRIQC_SLICES or edit SLICES_ROOT in the PATHS block.")
if mr.acq.eq("NONE").any():
    raise SystemExit(
        "Filenames carry no acquisition tag. A subject's standard and motion "
        "scans cannot be separated, so MR-ART must be re-extracted with "
        "sub-XXX_acq-YYY_sliceN.png naming.")

print(f"{len(mr)} slices | {mr.scan_id.nunique()} scans | "
      f"{mr.subject_id.nunique()} subjects")
print(f"  motion {mr[mr.y==1].scan_id.nunique()} | "
      f"acceptable {mr[mr.y==0].scan_id.nunique()}   (expect 198 / 129)")
both = (mr.groupby("subject_id").y.nunique() > 1).sum()
print(f"  subjects contributing BOTH classes: {both}")
print("  -> this is why scan-level splitting leaks; fine-tuning must split "
      "by subject_id")

# ---------------------------------------------------------------- predict
nrm = json.load(open(NORM))
assert nrm["positive_class"] == "BAD", "model polarity mismatch"

IMG = 128


def load_png(path, size=IMG):
    """MR-ART PNGs are stored at native resolution (e.g. 192x256) while the
    ABIDE v2 slices were resized to 128x128 at extraction. Resize here with
    bilinear interpolation, matching scipy zoom(order=1) used in 01."""
    im = Image.open(path).convert("L")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, np.float32)


sizes = {Image.open(p).size for p in mr.path.iloc[:50]}
print(f"\nsource PNG size(s): {sizes} -> resizing to ({IMG}, {IMG}) bilinear")

X = np.empty((len(mr), IMG, IMG, 1), np.float32)
for i, p in enumerate(mr.path):
    X[i, :, :, 0] = load_png(p)
    if i % 1000 == 0:
        print(f"  loaded {i}/{len(mr)}", end="\r")
X = (X - nrm["mean"]) / nrm["std"]          # ABIDE stats, deliberately not refit
print()

model = tf.keras.models.load_model(MODEL, compile=False)
mr["p"] = model.predict(X, batch_size=256, verbose=1).ravel()
mr.drop(columns=["path"]).to_csv(
    f"{RESULTS}/predictions/mrart_zeroshot_slice_probs.csv", index=False)

scan = mr.groupby(["scan_id", "subject_id", "y"]).p.mean().reset_index()
y, s, grp = scan.y.values, scan.p.values, scan.subject_id.values
prev = y.mean()
trivial_f1 = 2 * prev / (1 + prev)


def f1_at(yy, ss, t):
    _, _, f, _ = precision_recall_fscore_support(
        yy, (ss >= t).astype(int), average="binary", zero_division=0)
    return f


def boot(fn):
    u = np.unique(grp); idx = {g: np.where(grp == g)[0] for g in u}
    v = []
    for _ in range(N_BOOT):
        pick = RNG.choice(u, len(u), replace=True)
        i = np.concatenate([idx[g] for g in pick])
        if len(np.unique(y[i])) > 1:
            v.append(fn(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


auroc, auprc = roc_auc_score(y, s), average_precision_score(y, s)
lo_r, hi_r = boot(roc_auc_score)
lo_p, hi_p = boot(average_precision_score)
f1_05 = f1_at(y, s, 0.5)
cands = np.unique(np.round(s, 4))
f1_or, thr_or = max((f1_at(y, s, t), t) for t in cands)

print(f"\n=== zero-shot MR-ART ({len(scan)} scans, prevalence {prev:.3f}) ===")
print(f"  AUROC {auroc:.3f}  [{lo_r:.3f}, {hi_r:.3f}]")
print(f"  AUPRC {auprc:.3f}  [{lo_p:.3f}, {hi_p:.3f}]   chance {prev:.3f}")
print(f"  F1 @ inherited 0.5   {f1_05:.3f}")
print(f"  F1 @ oracle {thr_or:.3f}    {f1_or:.3f}")
print(f"  F1 of flag-everything {trivial_f1:.3f}  "
      f"(oracle beats it: {f1_or > trivial_f1 + 0.02})")
print(f"  recoverable by threshold alone: +{f1_or - f1_05:.3f}")
print("\n  AUROC ~0.5      -> signal did not transfer")
print("  AUROC high, F1 low -> operating-point failure, recoverable")

pd.DataFrame([{
    "model": "LightweightCNN", "n_scans": len(scan), "prevalence": prev,
    "AUROC": auroc, "AUROC_lo": lo_r, "AUROC_hi": hi_r,
    "AUPRC": auprc, "AUPRC_lo": lo_p, "AUPRC_hi": hi_p,
    "F1_inherited": f1_05, "F1_oracle": f1_or, "oracle_threshold": thr_or,
    "F1_trivial_all_positive": trivial_f1,
}]).to_csv(f"{RESULTS}/table2_mrart_zeroshot.csv", index=False)
scan.to_csv(f"{RESULTS}/predictions/mrart_zeroshot_scan_probs.csv", index=False)

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
b = np.linspace(0, 1, 41)
ax[0].hist(s[y == 0], bins=b, alpha=.6, label="acceptable", color="#4C72B0")
ax[0].hist(s[y == 1], bins=b, alpha=.6, label="motion", color="#C44E52")
ax[0].axvline(.5, color="k", ls="--", label="inherited 0.5")
ax[0].axvline(thr_or, color="green", ls=":", label=f"oracle {thr_or:.2f}")
ax[0].set(xlabel="P(motion-corrupted)", ylabel="scans",
          title="Score distribution")
ax[0].legend(fontsize=8)
fpr, tpr, _ = roc_curve(y, s)
ax[1].plot(fpr, tpr); ax[1].plot([0, 1], [0, 1], "k--", lw=.8)
ax[1].set(xlabel="FPR", ylabel="TPR", title=f"ROC (AUC {auroc:.3f})")
pr, rc, _ = precision_recall_curve(y, s)
ax[2].plot(rc, pr); ax[2].axhline(prev, color="k", ls="--", lw=.8)
ax[2].set(xlabel="Recall", ylabel="Precision", title=f"PR (AP {auprc:.3f})")
plt.tight_layout()
plt.savefig(f"{RESULTS}/figures/mrart_zeroshot.png", dpi=300)
print(f"\nwrote results to {RESULTS}")
