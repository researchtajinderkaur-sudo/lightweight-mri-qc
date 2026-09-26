"""
02_train_cnn.py  -  RUN IN COLAB (GPU)

Trains the lightweight CNN (25,857 params) on the rebuilt ABIDE cohort using
repeated stratified GROUP cross-validation.

Changes from the original training code, and why:
  * GroupKFold on subject, not a single 80/20 split
      -> all 48 BAD subjects serve as test cases, not 6
  * inner validation split for early stopping
      -> the original passed the TEST set as validation_data with
         restore_best_weights=True, selecting weights on test data
  * BAD = 1 (positive class), was GOOD = 1
      -> sigmoid now outputs P(motion-corrupted) directly, matching
         03_ml_and_confound.py. Downstream scripts assume this.
  * class_weight instead of subsampling the majority class
      -> subsampling by IQM values is what created the site confound
  * out-of-fold probabilities written to disk every fold
      -> so metrics never have to be recomputed from memory, and a
         disconnect loses at most one fold

Resumable: rerun after a disconnect and it skips completed folds.
"""

import os, re, glob, json, gc
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

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
SLICES  = os.path.join(SLICES_ROOT, "ABIDE_v2")
CKPT    = os.path.join(RESULTS, "cnn_folds")   # per-fold outputs, resumable

N_SPLITS, N_REPEATS = 5, 10
IMG, BATCH, EPOCHS, PATIENCE = 128, 64, 40, 10
VAL_FRAC = 0.15                               # of the training fold
SEED = 42
# ------------------------------------------------------------------------

os.makedirs(CKPT, exist_ok=True)
os.makedirs(f"{RESULTS}/predictions", exist_ok=True)
print("GPU:", tf.config.list_physical_devices("GPU"))

# ---------------------------------------------------------------- load
RX = re.compile(r"^(?P<site>[A-Za-z]+)_(?P<sub>\d+)_slice(?P<k>\d+)$")

rows = []
for cls, lab in (("GOOD", 0), ("BAD", 1)):        # BAD = positive
    for p in sorted(glob.glob(f"{SLICES}/{cls}/*.png")):
        m = RX.match(os.path.splitext(os.path.basename(p))[0])
        if not m:
            raise ValueError(f"unparsed filename: {p}")
        rows.append({"path": p, "y": lab, "site": m["site"],
                     "subject": f"{m['site']}_{m['sub']}"})
meta = pd.DataFrame(rows)

subj = meta.groupby("subject").agg(y=("y", "first"), site=("site", "first")).reset_index()
assert meta.groupby("subject").size().eq(20).all(), "not every subject has 20 slices"
print(f"{len(meta)} slices | {len(subj)} subjects | "
      f"BAD {int(subj.y.sum())} | GOOD {int((1-subj.y).sum())} | "
      f"prevalence {subj.y.mean():.4f}")

X = np.empty((len(meta), IMG, IMG, 1), np.float32)
for i, p in enumerate(meta.path):
    X[i, :, :, 0] = np.asarray(Image.open(p).convert("L"), np.float32)
    if i % 2000 == 0:
        print(f"  loaded {i}/{len(meta)}", end="\r")
y = meta.y.values
groups = meta.subject.values
print(f"\nloaded {X.shape}, {X.nbytes/1e9:.2f} GB")


def build():
    """Original architecture. 25,857 trainable parameters."""
    tf.keras.backend.clear_session()
    m = tf.keras.Sequential([
        tf.keras.layers.Input((IMG, IMG, 1)),
        tf.keras.layers.Conv2D(16, 3, padding="same"),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same"),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same"),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m


print(f"\nparams: {build().count_params():,}  (expect 25,857)\n")

# ---------------------------------------------------------------- CV
sy, sg = subj.y.values, subj.subject.values
w = {0: 1.0, 1: float((sy == 0).sum() / (sy == 1).sum())}
print(f"class_weight: {w[1]:.1f}:1\n")

for rep in range(N_REPEATS):
    cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED + rep)
    for fold, (tr_i, te_i) in enumerate(cv.split(subj, sy, groups=sg)):
        tag = f"rep{rep:02d}_fold{fold}"
        out = f"{CKPT}/{tag}.csv"
        if os.path.exists(out):
            print(f"{tag}: done, skipping")
            continue

        tr_s, te_s = set(sg[tr_i]), set(sg[te_i])
        assert not (tr_s & te_s), "subject leak between train and test"

        # inner validation split, still subject-disjoint
        i_tr, i_va = train_test_split(sorted(tr_s), test_size=VAL_FRAC,
                                      random_state=SEED,
                                      stratify=subj.set_index("subject")
                                      .loc[sorted(tr_s), "y"].values)
        i_tr, i_va = set(i_tr), set(i_va)

        m_tr = np.isin(groups, list(i_tr))
        m_va = np.isin(groups, list(i_va))
        m_te = np.isin(groups, list(te_s))

        mu, sd = X[m_tr].mean(), X[m_tr].std()      # train-fold stats only
        norm = lambda a: (a - mu) / sd

        model = build()
        model.fit(norm(X[m_tr]), y[m_tr],
                  validation_data=(norm(X[m_va]), y[m_va]),
                  epochs=EPOCHS, batch_size=BATCH, class_weight=w, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                                 "val_loss", patience=PATIENCE,
                                 restore_best_weights=True),
                             tf.keras.callbacks.ReduceLROnPlateau(
                                 "val_loss", factor=0.5, patience=5)])

        p = model.predict(norm(X[m_te]), batch_size=256, verbose=0).ravel()
        df = pd.DataFrame({"subject": groups[m_te], "y": y[m_te], "p": p})
        agg = df.groupby(["subject", "y"]).p.mean().reset_index()
        agg["rep"], agg["fold"], agg["mu"], agg["sd"] = rep, fold, mu, sd
        agg.to_csv(out, index=False)

        print(f"{tag}: n_test={len(agg)} BAD={int(agg.y.sum())} "
              f"AUROC={roc_auc_score(agg.y, agg.p):.3f} "
              f"AUPRC={average_precision_score(agg.y, agg.p):.3f}")
        del model; gc.collect()

# ---------------------------------------------------------------- pool
files = sorted(glob.glob(f"{CKPT}/rep*_fold*.csv"))
print(f"\npooling {len(files)} folds")
allf = pd.concat([pd.read_csv(f) for f in files])
oof = allf.groupby(["subject", "y"]).p.mean().reset_index()
oof = oof.merge(subj[["subject", "site"]], on="subject")
oof.to_csv(f"{RESULTS}/predictions/abide_cnn_oof_probs.csv", index=False)

per = (allf.groupby(["rep", "fold"])
       .apply(lambda g: pd.Series({
           "AUROC": roc_auc_score(g.y, g.p),
           "AUPRC": average_precision_score(g.y, g.p)}))
       .reset_index())

prev = oof.y.mean()
summary = pd.DataFrame([{
    "model": "LightweightCNN",
    "AUROC_pooled": roc_auc_score(oof.y, oof.p),
    "AUPRC_pooled": average_precision_score(oof.y, oof.p),
    "AUROC_mean": per.AUROC.mean(), "AUROC_sd": per.AUROC.std(),
    "AUPRC_mean": per.AUPRC.mean(), "AUPRC_sd": per.AUPRC.std(),
    "chance_AUPRC": prev, "n_subjects": len(oof), "n_BAD": int(oof.y.sum()),
}])
print("\n=== CNN, pooled out-of-fold ===")
print(summary.round(3).to_string(index=False))
summary.to_csv(f"{RESULTS}/table1_cnn.csv", index=False)

# save one model trained on everything, for MR-ART transfer
mu, sd = X.mean(), X.std()
final = build()
tr_s, va_s = train_test_split(subj.subject.values, test_size=VAL_FRAC,
                              random_state=SEED, stratify=sy)
m_tr, m_va = np.isin(groups, tr_s), np.isin(groups, va_s)
final.fit((X[m_tr]-mu)/sd, y[m_tr],
          validation_data=((X[m_va]-mu)/sd, y[m_va]),
          epochs=EPOCHS, batch_size=BATCH, class_weight=w, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(
              "val_loss", patience=PATIENCE, restore_best_weights=True)])
os.makedirs(f"{V4}/models", exist_ok=True)
final.save(f"{V4}/models/abide_cnn_v4.keras")
json.dump({"mean": float(mu), "std": float(sd), "positive_class": "BAD",
           "n_subjects": int(len(subj)), "n_bad": int(sy.sum())},
          open(f"{V4}/models/abide_cnn_v4_norm.json", "w"), indent=2)
print(f"\nsaved model + norm stats (mean={mu:.2f}, std={sd:.2f}) to {V4}/models/")
