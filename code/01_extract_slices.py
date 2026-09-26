"""
01_extract_slices.py  -  RUN LOCALLY (laptop, mri_clean_env)

Extracts 20 centered axial slices per subject from the rebuilt ABIDE cohort
and writes 8-bit grayscale PNGs named SITE_SUBID_sliceN.png.

Uses PIL, not plt.imsave. plt.imsave writes RGBA and applies a colormap; that
would silently change the model input relative to the original extraction.

Set TEST_MODE = True first. Verify 10 subjects, then set it False.
"""

import os, re, glob
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
from scipy.ndimage import zoom

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
GOOD_CSV   = os.path.join(COHORT, "good_subjects_rebuilt.csv")
BAD_CSV    = os.path.join(COHORT, "bad_subjects_rebuilt.csv")
# Downloaded ABIDE I NIfTI tree: <ABIDE_ROOT>/<SITE_SUBID>/anat/NIfTI/mprage.nii.gz
ABIDE_ROOT = os.environ.get("MRIQC_ABIDE_NIFTI", os.path.join(ROOT, "data", "ABIDE"))
OUT_DIR    = os.path.join(SLICES_ROOT, "ABIDE_v2")
MANIFEST   = os.path.join(COHORT, "extraction_manifest.csv")

N_SLICES  = 20
IMG_SIZE  = 128
TEST_MODE = True          # True -> first 5 GOOD + 5 BAD only
# ------------------------------------------------------------------------


def find_nifti(scan_id):
    """ABIDE tree: ABIDE/<SITE_SUBID>/anat/NIfTI/mprage.nii.gz, with fallbacks.

    scan_id is sanitized (MAXMUN_51333) so it parses as exactly SITE_SUBID.
    The folder on disk may still use the original SITE_ID (MAX_MUN_51333),
    so try both.
    """
    site, sub = scan_id.split("_")
    candidates = [scan_id]
    if site == "MAXMUN":
        candidates.append(f"MAX_MUN_{sub}")
    for folder in candidates:
        base = os.path.join(ABIDE_ROOT, folder)
        for c in [os.path.join(base, "anat", "NIfTI", "mprage.nii.gz"),
                  os.path.join(base, "anat", "NIfTI", "mprage.nii"),
                  os.path.join(base, "anat", "mprage.nii.gz"),
                  os.path.join(base, "anat", "NIfTI", f"{folder}_anat_mprage.nii.gz"),
                  os.path.join(base, f"{folder}.nii.gz")]:
            if os.path.exists(c):
                return c
        hits = glob.glob(os.path.join(base, "**", "*.nii*"), recursive=True)
        if hits:
            return hits[0]
    return None


def extract(nifti_path, out_dir, scan_id):
    """20 centered axial slices -> per-slice min-max [0,255] -> 128x128 -> PNG."""
    vol = np.asarray(nib.load(nifti_path).dataobj).astype(np.float32)
    if vol.ndim != 3:
        vol = np.squeeze(vol)
    if vol.ndim != 3:
        return 0, "not 3D"

    mid = vol.shape[2] // 2
    lo = max(0, mid - N_SLICES // 2)
    hi = min(vol.shape[2], lo + N_SLICES)
    lo = max(0, hi - N_SLICES)
    if hi - lo < N_SLICES:
        return 0, f"only {hi-lo} slices available"

    written = 0
    for i, z in enumerate(range(lo, hi)):
        sl = vol[:, :, z]
        rng = sl.max() - sl.min()
        sl = (sl - sl.min()) / rng * 255.0 if rng > 0 else np.zeros_like(sl)

        f = (IMG_SIZE / sl.shape[0], IMG_SIZE / sl.shape[1])
        sl = zoom(sl, f, order=1)
        if sl.shape != (IMG_SIZE, IMG_SIZE):
            pad = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
            h, w = min(IMG_SIZE, sl.shape[0]), min(IMG_SIZE, sl.shape[1])
            pad[:h, :w] = sl[:h, :w]
            sl = pad

        # mode 'L' = 8-bit grayscale, matching the original extraction
        Image.fromarray(sl.astype(np.uint8), mode="L").save(
            os.path.join(out_dir, f"{scan_id}_slice{i}.png"))
        written += 1
    return written, None


good = pd.read_csv(GOOD_CSV).scan_id.astype(str).tolist()
bad  = pd.read_csv(BAD_CSV).scan_id.astype(str).tolist()
if TEST_MODE:
    good, bad = good[:5], bad[:5]
print(f"cohort: {len(good)} GOOD, {len(bad)} BAD"
      f"{'   [TEST MODE]' if TEST_MODE else ''}")

for cls in ("GOOD", "BAD"):
    os.makedirs(os.path.join(OUT_DIR, cls), exist_ok=True)

rows = []
for cls, ids in (("GOOD", good), ("BAD", bad)):
    out_dir = os.path.join(OUT_DIR, cls)
    for k, sid in enumerate(ids, 1):
        p = find_nifti(sid)
        if p is None:
            rows.append({"scan_id": sid, "cls": cls, "n_slices": 0,
                         "status": "no NIfTI found", "path": ""})
            print(f"  [{k}/{len(ids)}] {sid}: NOT FOUND")
            continue
        try:
            n, err = extract(p, out_dir, sid)
            rows.append({"scan_id": sid, "cls": cls, "n_slices": n,
                         "status": err or "ok", "path": p})
            print(f"  [{k}/{len(ids)}] {sid}: {n} slices"
                  + (f"  ({err})" if err else ""))
        except Exception as e:
            rows.append({"scan_id": sid, "cls": cls, "n_slices": 0,
                         "status": f"error: {e}", "path": p})
            print(f"  [{k}/{len(ids)}] {sid}: ERROR {e}")

man = pd.DataFrame(rows)
os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
man.to_csv(MANIFEST, index=False)

print(f"\n{'='*60}")
ok = man[man.n_slices == N_SLICES]
print(f"complete ({N_SLICES} slices): {len(ok)} / {len(man)} subjects")
print(f"  GOOD {int((ok.cls=='GOOD').sum())}   BAD {int((ok.cls=='BAD').sum())}")
bad_rows = man[man.n_slices != N_SLICES]
if len(bad_rows):
    print(f"\nincomplete or failed: {len(bad_rows)}")
    print(bad_rows[["scan_id", "cls", "n_slices", "status"]].to_string(index=False))
    print("\n-> record these in Methods as exclusions")
print(f"\nmanifest: {MANIFEST}")

# verification
sample = glob.glob(os.path.join(OUT_DIR, "GOOD", "*.png"))
if sample:
    im = Image.open(sample[0])
    print(f"\nsample: {os.path.basename(sample[0])}  mode={im.mode}  size={im.size}")
    if im.mode != "L":
        print("  *** WARNING: expected mode 'L' (8-bit grayscale). "
              "RGBA means a colormap was applied and the model input differs. ***")
    else:
        print("  grayscale confirmed")

if TEST_MODE:
    print("\nTEST MODE was on. Check names and mode above, then set "
          "TEST_MODE = False and rerun for all 790.")
