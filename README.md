# lightweight-mri-qc
# Lightweight CNN for Brain MRI Motion Artifact Detection

Code and cohort definitions for:

> **Accuracy–Efficiency Trade-offs in Cross-Dataset Brain MRI Motion Artifact
> Detection: A 26,000-Parameter CNN Generalises With Minimal Degradation**
> Tajinder Kaur, Sikander Singh Cheema, Lakhwinder Kaur. *Magnetic Resonance Imaging*, submitted.

A convolutional network with **25,857 trainable parameters** that detects motion
artifacts in T1-weighted structural brain MRI, evaluated in-domain on ABIDE I and
zero-shot on the independent MR-ART dataset.

---

## Headline results

| | AUROC | AUPRC | Chance AUPRC |
|---|---|---|---|
| ABIDE, in-domain (5×10 repeated grouped CV) | 0.898 ± 0.050 | 0.644 ± 0.141 | 0.061 |
| MR-ART, zero-shot (no adaptation) | 0.897 (95% CI 0.864–0.929) | 0.936 | 0.606 |

Discrimination transfers with no measurable loss across scanner, acquisition
protocol and label definition. Threshold-dependent performance does not: at the
inherited 0.5 threshold F1 is 0.717, below the 0.754 obtained by flagging every
scan. Selecting a threshold from **10 labelled target subjects**, with weights
frozen, raises F1 to 0.821 ± 0.049.

Site identity alone, with no image data, predicts ABIDE quality labels at
AUROC 0.835 — which is why we report a site-only control baseline.

---

## Reproducibility

Every script writes per-fold predicted probabilities to disk **before** any
metric is computed, so every value in the paper can be recomputed from stored
predictions without retraining. Figures are generated from the results CSVs;
no number is hardcoded in the plotting code.

---

## Data

Neither dataset is redistributed here. Both are publicly available:

- **ABIDE I** — http://fcon_1000.projects.nitrc.org/indi/abide/
- **MR-ART** — https://openneuro.org/datasets/ds004173

`cohort/` contains the subject lists needed to reproduce the exact cohort.

### ABIDE cohort construction

Labels come **only** from the manual quality assessments distributed with ABIDE.
No image quality metric was used at any point in label construction — selecting
scans on metrics that later serve as classifier inputs induces circularity, and
because those metrics carry scanner-specific signatures it also concentrates
classes within sites.

| Rater 2 | Rater 3 | Label | n |
|---|---|---|---|
| OK | OK | acceptable | 784 |
| fail | fail | motion-corrupted | 5 |
| maybe | fail | motion-corrupted | 46 |
| any other combination | | excluded | |

835 labelled scans → 45 had no accessible NIfTI → 1 excluded for non-standard
dimensionality → **789 subjects (741 acceptable, 48 motion-corrupted;
prevalence 6.08%)** across 17 sites, 10 of which contain both classes.

### MR-ART cohort

Motion score 1 → acceptable, score 3 → motion-corrupted, score 2 excluded
(dataset convention). **327 scans (129 / 198) from 144 subjects.**

100 of the 144 subjects contribute scans of *both* classes, so all MR-ART
partitioning is done at subject level. Scan-level splitting places the same
anatomy on both sides.

---

## Pipeline

| Script | Runs on | Produces |
|---|---|---|
| `01_extract_slices.py` | CPU | 20 centered axial slices/subject, 128×128 8-bit PNG |
| `02_train_cnn.py` | GPU | CNN, 5×10 grouped CV, out-of-fold probabilities |
| `03_ml_and_confound.py` | CPU, seconds | IQM baselines, site-only control, leave-one-site-out |
| `04b_architecture_comparison.py` | GPU | MobileNetV3/EfficientNet/DenseNet/ResNet under matched protocol |
| `05_mrart_zeroshot.py` | GPU | MR-ART zero-shot AUROC/AUPRC, bootstrap CIs, threshold analysis |
| `06_recalibration.py` | CPU, seconds | label-budget recalibration curve |
| `07_figures.py` | CPU | all figures, from results CSVs only |

Each script has a `CONFIG` block at the top; set the paths there before running.

Scripts 02 and 04b are resumable — completed folds are written to disk and
skipped on rerun.

---

## Model

Three convolutional blocks (16, 32, 64 filters; 3×3, same padding), each with
batch normalisation, ReLU and 2×2 max pooling; global average pooling; a 32-unit
dense layer; dropout (p = 0.6); single sigmoid output. **25,857 parameters.**

Global average pooling in place of flattening removes the dense layer that
dominates parameter counts in conventional architectures, while giving
translational tolerance suited to artifacts distributed across the image.

**Positive class is motion-corrupted**, so the sigmoid outputs
P(motion-corrupted) directly.

---

## Evaluation notes

- **Partitioning is by subject**, never by slice or scan. All 20 slices from a
  subject fall in one fold.
- **Early stopping uses an inner validation split** drawn from the training
  fold. The test fold is never used for model selection or monitoring.
- **AUPRC is interpreted against class prevalence** (0.061 ABIDE, 0.606 MR-ART).
  At 6% prevalence a default 0.5 threshold is not meaningful — one baseline here
  reaches AUPRC 5.9× chance with F1 = 0.000.
- **Confidence intervals use subject-level cluster bootstrap.** Resampling scans
  independently understates uncertainty because acquisitions from one subject
  are correlated. No resamples are discarded on the basis of predicted values.
- **MR-ART is never used for training, architecture selection or
  hyperparameters**, except in the recalibration analysis where the labelled
  budget is stated explicitly.

---

## Requirements

```
tensorflow>=2.15
scikit-learn
numpy, pandas, matplotlib
pillow, nibabel, scipy
openpyxl
```

Training used an NVIDIA A100. Scripts 03, 06 and 07 need no GPU.

---

## Citation

```bibtex
@article{kaur_lightweight_mri_qc,
  title   = {Accuracy--Efficiency Trade-offs in Cross-Dataset Brain MRI Motion
             Artifact Detection: A 26,000-Parameter CNN Generalises With
             Minimal Degradation},
  author  = {Tajinder Kaur, Sikander Singh Cheema and Lakhwinder Kaur},
  journal = {Magnetic Resonance Imaging},
  year    = {2026},
  note     = {submitted}
}
```

## License

Code released under the MIT License. The datasets retain their own terms of use.
