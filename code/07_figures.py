"""
07_figures.py  -  runs anywhere, seconds, no GPU

Generates every figure in the paper from the CSVs in results/.
No value is typed into this script. If a number changes upstream, rerun this
and the figures change with it.

Figures:
  fig1_score_distributions.png   MR-ART scores vs both thresholds
  fig2_roc_pr.png                ROC and PR, ABIDE in-domain vs MR-ART zero-shot
  fig3_confusion_matrices.png    same predictions at 0.5 and at recalibrated
  fig4_recalibration.png         F1 vs label budget
  fig5_params_vs_auroc.png       accuracy-efficiency frontier
  fig6_site_confound.png         site-only control + leave-one-site-out
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score, confusion_matrix)

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
R    = RESULTS
FIG  = os.path.join(RESULTS, "figures")
DPI  = 300
# ------------------------------------------------------------------------

os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110})
C_OK, C_BAD, C_ACC = "#4C72B0", "#C44E52", "#55A868"

mr = pd.read_csv(f"{R}/predictions/mrart_zeroshot_scan_probs.csv")
ym, sm = mr.y.values.astype(int), mr.p.values
recal = pd.read_csv(f"{R}/table3_recalibration.csv")
THR = float(recal.loc[recal.N == 10, "threshold_median"].iloc[0])
prev = ym.mean()

# ---- fig 1: score distributions ---------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.6))
b = np.linspace(0, 1, 41)
ax.hist(sm[ym == 0], bins=b, alpha=.65, color=C_OK, label="acceptable (score 1)")
ax.hist(sm[ym == 1], bins=b, alpha=.65, color=C_BAD, label="motion (score 3)")
ax.axvline(0.5, color="k", ls="--", lw=1.4, label="inherited threshold (0.50)")
ax.axvline(THR, color=C_ACC, ls="-", lw=1.8,
           label=f"recalibrated (n=10) ({THR:.2f})")
ax.set(xlabel="P(motion-corrupted)", ylabel="number of scans")
ax.legend(fontsize=7.5, frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_score_distributions.png", dpi=DPI)
plt.close(fig)

# ---- fig 2: ROC + PR, both domains ------------------------------------
ab_path = f"{R}/predictions/abide_cnn_oof_probs.csv"
have_ab = os.path.exists(ab_path)
if have_ab:
    ab = pd.read_csv(ab_path)
    ya, sa = ab.y.values.astype(int), ab.p.values

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.9))
if have_ab:
    f, t, _ = roc_curve(ya, sa)
    a1.plot(f, t, color=C_OK, label=f"ABIDE in-domain ({roc_auc_score(ya,sa):.3f})")
f, t, _ = roc_curve(ym, sm)
a1.plot(f, t, color=C_BAD, label=f"MR-ART zero-shot ({roc_auc_score(ym,sm):.3f})")
a1.plot([0, 1], [0, 1], "k--", lw=.8)
a1.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC")
a1.legend(fontsize=7.5, frameon=False, loc="lower right")

if have_ab:
    p, r, _ = precision_recall_curve(ya, sa)
    a2.plot(r, p, color=C_OK,
            label=f"ABIDE ({average_precision_score(ya,sa):.3f}, chance {ya.mean():.3f})")
    a2.axhline(ya.mean(), color=C_OK, ls=":", lw=.8)
p, r, _ = precision_recall_curve(ym, sm)
a2.plot(r, p, color=C_BAD,
        label=f"MR-ART ({average_precision_score(ym,sm):.3f}, chance {prev:.3f})")
a2.axhline(prev, color=C_BAD, ls=":", lw=.8)
a2.set(xlabel="recall", ylabel="precision", title="Precision-Recall")
a2.legend(fontsize=7.5, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_roc_pr.png", dpi=DPI); plt.close(fig)

# ---- fig 3: paired confusion matrices ---------------------------------
def draw_cm(ax, thr, title):
    cm = confusion_matrix(ym, (sm >= thr).astype(int))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() * .55 else "black")
    tn, fp, fn, tp = cm.ravel()
    rec = tp / (tp + fn); pre = tp / (tp + fp) if tp + fp else 0
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["pred\nacceptable", "pred\nmotion"],
           yticklabels=["true\nacceptable", "true\nmotion"])
    ax.set_title(f"{title}\nF1 {f1:.3f}  precision {pre:.3f}  recall {rec:.3f}",
                 fontsize=8.5)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.9))
draw_cm(a1, 0.5, "Inherited threshold (0.50)")
draw_cm(a2, THR, f"Recalibrated, 10 subjects ({THR:.2f})")
fig.suptitle("Identical model and predictions; only the threshold differs",
             fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_confusion_matrices.png", dpi=DPI)
plt.close(fig)

# ---- fig 4: recalibration curve ---------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(recal.N, recal.F1_mean, yerr=recal.F1_sd, fmt="o-", color=C_ACC,
            capsize=3, lw=1.8, label="recalibrated (weights frozen)")
ax.fill_between(recal.N, recal.F1_p10, recal.F1_mean + recal.F1_sd,
                color=C_ACC, alpha=.12)
ax.axhline(recal.F1_oracle.mean(), ls=":", color="k", lw=1,
           label=f"oracle threshold ({recal.F1_oracle.mean():.3f})")
ax.axhline(recal.F1_at_05.mean(), ls="--", color=C_BAD, lw=1,
           label=f"inherited 0.50 ({recal.F1_at_05.mean():.3f})")
ax.axhline(recal.F1_flag_all.iloc[0], ls="-.", color="gray", lw=1,
           label=f"flag every scan ({recal.F1_flag_all.iloc[0]:.3f})")
ax.set(xlabel="labelled target subjects used to set threshold",
       ylabel="F1 on held-out scans", xticks=recal.N)
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_recalibration.png", dpi=DPI)
plt.close(fig)

# ---- fig 5: params vs AUROC -------------------------------------------
t4 = pd.read_csv(f"{R}/table4_architectures.csv")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for _, r_ in t4.iterrows():
    ours = "ours" in r_.arch.lower()
    ax.errorbar(r_.params, r_.mrart_AUROC_mean, yerr=r_.mrart_AUROC_sd,
                fmt="*" if ours else "o", ms=18 if ours else 8,
                color=C_BAD if ours else C_OK, capsize=3, zorder=3)
    ax.annotate(r_.arch.replace(" (ours)", ""),
                (r_.params, r_.mrart_AUROC_mean),
                textcoords="offset points", xytext=(0, 14 if ours else 10),
                ha="center", fontsize=7.5,
                fontweight="bold" if ours else "normal")
ax.set_xscale("log")
ax.set(xlabel="trainable parameters (log scale)",
       ylabel="zero-shot MR-ART AUROC")
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_params_vs_auroc.png", dpi=DPI)
plt.close(fig)

# ---- fig 6: site confound + LOSO --------------------------------------
sc = pd.read_csv(f"{R}/site_confound.csv")
lo = pd.read_csv(f"{R}/leave_one_site_out.csv").dropna(subset=["AUROC"])
# ABIDE chance AUPRC = class prevalence, read from the cohort, never typed in
ab_prev = (pd.read_csv(ab_path).y.mean() if have_ab
           else lo.n_BAD.sum() / lo.n.sum())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4),
                             gridspec_kw={"width_ratios": [1, 1.25]})
x = np.arange(len(sc))
a1.bar(x - .2, sc.AUROC, .4, label="AUROC", color=C_OK)
a1.bar(x + .2, sc.AUPRC, .4, label="AUPRC", color=C_BAD)
a1.axhline(ab_prev, ls=":", color="gray", lw=1,
           label=f"chance AUPRC ({ab_prev:.3f})")
a1.set(xticks=x, ylabel="score", title="What does site alone explain?")
a1.set_xticklabels(sc.features, rotation=15, ha="right", fontsize=8)
a1.legend(fontsize=7.5, frameon=False)

lo = lo.sort_values("AUROC")
a2.barh(lo.site, lo.AUROC, color=C_OK)
a2.axvline(0.5, ls="--", color="k", lw=.9, label="chance")
a2.axvline(lo.AUROC.mean(), ls=":", color=C_BAD, lw=1.2,
           label=f"mean {lo.AUROC.mean():.3f}")
a2.set(xlabel="AUROC on held-out site", xlim=(0, 1),
       title="Leave-one-site-out")
a2.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/fig6_site_confound.png", dpi=DPI)
plt.close(fig)

print(f"wrote 6 figures to {FIG}")
for f in sorted(os.listdir(FIG)):
    print("  ", f)
