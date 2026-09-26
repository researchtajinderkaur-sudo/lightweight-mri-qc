"""
04b_architecture_comparison.py  -  RUN IN COLAB (GPU)

Tests the paper's central claim: does 25,857 parameters suffice, or do larger
architectures do better?

Every model sees identical data, identical subject-disjoint folds, identical
class weighting, and is evaluated both in-domain (ABIDE) and zero-shot
(MR-ART). Differences are therefore attributable to architecture alone.

Also measures CPU latency and model size, so the efficiency claim rests on
numbers rather than parameter count alone.

Runtime: ~2-3 h on A100 at N_REPEATS=3. Resumable - completed runs are skipped.
"""

import os, re, glob, json, time, gc
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
ABIDE   = os.path.join(SLICES_ROOT, "ABIDE_v2")
MRART   = os.path.join(SLICES_ROOT, "MR_ART_SLICES")
CKPT    = os.path.join(RESULTS, "arch_folds")

N_SPLITS, N_REPEATS = 5, 3
IMG, BATCH, EPOCHS, PATIENCE = 128, 64, 40, 10
VAL_FRAC, SEED = 0.15, 42
# ------------------------------------------------------------------------

os.makedirs(CKPT, exist_ok=True)
print("GPU:", tf.config.list_physical_devices("GPU"))


def load_folder(root, rx, labelled):
    rows = []
    for cls, lab in labelled:
        for p in sorted(glob.glob(f"{root}/{cls}/*.png")):
            m = rx.match(os.path.splitext(os.path.basename(p))[0])
            if not m:
                raise ValueError(f"unparsed: {p}")
            rows.append({"path": p, "y": lab, **m.groupdict()})
    if not rows:
        raise SystemExit(f"no PNGs under {root} - check the path in CONFIG")
    return pd.DataFrame(rows)


def load_images(paths, size=IMG):
    X = np.empty((len(paths), size, size, 1), np.float32)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("L")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        X[i, :, :, 0] = np.asarray(im, np.float32)
    return X


# ---- data -------------------------------------------------------------
ab = load_folder(ABIDE, re.compile(r"^(?P<site>[A-Za-z]+)_(?P<sub>\d+)_slice\d+$"),
                 [("GOOD", 0), ("BAD", 1)])
ab["subject"] = ab.site + "_" + ab["sub"]
Xa, ya, ga = load_images(ab.path.tolist()), ab.y.values, ab.subject.values

mr = load_folder(MRART,
                 re.compile(r"^(?P<sid>sub-[A-Za-z0-9]+)_(?P<acq>acq-[A-Za-z0-9]+)_slice\d+$"),
                 [("GOOD", 0), ("BAD", 1)])
mr["scan_id"] = mr.sid + "_" + mr.acq
Xm, ym = load_images(mr.path.tolist()), mr.y.values

subj = ab.groupby("subject").y.first().reset_index()
sy, sg = subj.y.values, subj.subject.values
W = {0: 1.0, 1: float((sy == 0).sum() / (sy == 1).sum())}
print(f"ABIDE {Xa.shape} | MR-ART {Xm.shape} | class_weight {W[1]:.1f}:1")


# ---- architectures ----------------------------------------------------
# All models receive RAW [0,255] input and normalise internally.
# This matters: feeding z-scored data into an ImageNet backbone and then
# applying its preprocess_input double-normalises and destroys the input
# scale the pretrained weights expect.
AB_MEAN, AB_STD = None, None      # set from the training fold below


def lightweight_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Input((IMG, IMG, 1)),
        tf.keras.layers.Normalization(mean=AB_MEAN, variance=AB_STD ** 2),
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


def pretrained(fn, prep):
    """ImageNet-pretrained backbone fed raw [0,255], tiled to 3 channels,
    then passed through the backbone's own preprocess_input."""
    inp = tf.keras.layers.Input((IMG, IMG, 1))
    x = tf.keras.layers.Concatenate()([inp, inp, inp])
    x = tf.keras.layers.Lambda(prep, output_shape=(IMG, IMG, 3))(x)
    base = fn(include_top=False, weights="imagenet",
              input_shape=(IMG, IMG, 3), pooling="avg")
    x = base(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)


ka = tf.keras.applications
ARCHS = {
    "LightweightCNN (ours)": lightweight_cnn,
    "MobileNetV3Small": lambda: pretrained(ka.MobileNetV3Small,
                                           ka.mobilenet_v3.preprocess_input),
    "EfficientNetB0":   lambda: pretrained(ka.EfficientNetB0,
                                           ka.efficientnet.preprocess_input),
    "DenseNet121":      lambda: pretrained(ka.DenseNet121,
                                           ka.densenet.preprocess_input),
    "ResNet50":         lambda: pretrained(ka.ResNet50,
                                           ka.resnet.preprocess_input),
}

# Pretrained backbones need a smaller LR: 1e-3 with Adam wipes out ImageNet
# weights within a few steps.
LR = {"LightweightCNN (ours)": 1e-3}


def build(name):
    tf.keras.backend.clear_session()
    m = ARCHS[name]()
    m.compile(optimizer=tf.keras.optimizers.Adam(LR.get(name, 1e-4)),
              loss="binary_crossentropy")
    return m


AB_MEAN, AB_STD = float(Xa.mean()), float(Xa.std())   # placeholder for the
print("\nparameter counts:")                              # param-count pass
for n in ARCHS:
    print(f"  {n:24s} {build(n).count_params():>12,}  lr={LR.get(n,1e-4):.0e}")

# ---- CV ---------------------------------------------------------------
for name in ARCHS:
    slug = re.sub(r"\W+", "_", name).strip("_")
    for rep in range(N_REPEATS):
        cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED + rep)
        for fold, (tr_i, te_i) in enumerate(cv.split(subj, sy, groups=sg)):
            out = f"{CKPT}/{slug}_rep{rep}_fold{fold}.json"
            if os.path.exists(out):
                continue

            tr_s, te_s = set(sg[tr_i]), set(sg[te_i])
            i_tr, i_va = train_test_split(
                sorted(tr_s), test_size=VAL_FRAC, random_state=SEED,
                stratify=subj.set_index("subject").loc[sorted(tr_s), "y"].values)
            m_tr = np.isin(ga, list(i_tr)); m_va = np.isin(ga, list(i_va))
            m_te = np.isin(ga, list(te_s))

            # fold statistics go INSIDE the model; all models see raw [0,255]
            globals()["AB_MEAN"] = float(Xa[m_tr].mean())
            globals()["AB_STD"] = float(Xa[m_tr].std())

            model = build(name)
            t0 = time.time()
            model.fit(Xa[m_tr], ya[m_tr],
                      validation_data=(Xa[m_va], ya[m_va]),
                      epochs=EPOCHS, batch_size=BATCH, class_weight=W, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(
                          "val_loss", patience=PATIENCE, restore_best_weights=True)])
            train_s = time.time() - t0

            # in-domain, subject level
            p = model.predict(Xa[m_te], batch_size=256, verbose=0).ravel()
            d = pd.DataFrame({"s": ga[m_te], "y": ya[m_te], "p": p})
            agg = d.groupby(["s", "y"]).p.mean().reset_index()

            # zero-shot MR-ART, scan level
            pm = model.predict(Xm, batch_size=256, verbose=0).ravel()
            dm = pd.DataFrame({"s": mr.scan_id.values, "y": ym, "p": pm})
            aggm = dm.groupby(["s", "y"]).p.mean().reset_index()

            rec = {"arch": name, "params": int(model.count_params()),
                   "rep": rep, "fold": fold, "train_s": train_s,
                   "lr": LR.get(name, 1e-4),
                   "abide_AUROC": roc_auc_score(agg.y, agg.p),
                   "abide_AUPRC": average_precision_score(agg.y, agg.p),
                   "mrart_AUROC": roc_auc_score(aggm.y, aggm.p),
                   "mrart_AUPRC": average_precision_score(aggm.y, aggm.p)}
            json.dump(rec, open(out, "w"))
            print(f"{name[:20]:20s} r{rep}f{fold}  "
                  f"ABIDE {rec['abide_AUROC']:.3f} | "
                  f"MR-ART {rec['mrart_AUROC']:.3f}")
            del model; gc.collect()

# ---- CPU latency ------------------------------------------------------
print("\nmeasuring CPU latency (20 slices = 1 subject)")
AB_MEAN, AB_STD = float(Xa.mean()), float(Xa.std())
lat = {}
with tf.device("/CPU:0"):
    for name in ARCHS:
        m = build(name)
        x = np.full((20, IMG, IMG, 1), 128.0, np.float32)   # raw [0,255] scale
        m.predict(x, verbose=0)
        ts = [time.time() for _ in range(1)]
        t0 = time.time()
        for _ in range(10):
            m.predict(x, verbose=0)
        lat[name] = (time.time() - t0) / 10 * 1000
        print(f"  {name:24s} {lat[name]:8.1f} ms/subject")
        del m; gc.collect()

# ---- summary ----------------------------------------------------------
recs = [json.load(open(f)) for f in glob.glob(f"{CKPT}/*.json")]
df = pd.DataFrame(recs)
summ = (df.groupby("arch")
        .agg(params=("params", "first"),
             abide_AUROC_mean=("abide_AUROC", "mean"),
             abide_AUROC_sd=("abide_AUROC", "std"),
             mrart_AUROC_mean=("mrart_AUROC", "mean"),
             mrart_AUROC_sd=("mrart_AUROC", "std"),
             mrart_AUPRC_mean=("mrart_AUPRC", "mean"),
             train_s=("train_s", "mean"))
        .reset_index())
summ["cpu_ms_per_subject"] = summ.arch.map(lat)
summ["params_vs_ours"] = (summ.params / summ.params.min()).round(1)
summ = summ.sort_values("mrart_AUROC_mean", ascending=False)

print("\n=== architecture comparison ===")
print(summ.round(3).to_string(index=False))
summ.to_csv(f"{RESULTS}/table4_architectures.csv", index=False)
df.to_csv(f"{RESULTS}/table4_architectures_raw.csv", index=False)
print(f"\nwrote {RESULTS}/table4_architectures.csv")
