# Lightweight CNN for Brain MRI Quality Assessment

A 25,857-parameter convolutional network for binary quality control of
structural brain MRI, trained on ABIDE I and evaluated zero-shot on MR-ART.

**Paper:** Kaur T, Cheema SS, Kaur L. Cross-Dataset Structural Brain MRI
Quality Assessment with a 25,857-Parameter CNN: Capacity Is Not the Limiting
Factor. *Magnetic Resonance Imaging*. 2026. Accepted.

---

## The model

| | |
|---|---|
| Total parameters | 25,857 |
| Trainable parameters | 25,633 |
| Non-trainable | 224 (batch-normalisation running statistics) |
| Input | 20 axial slices per volume, 128 × 128, normalised to [0, 255] |
| Architecture | 3 × (conv 3×3 → batch norm → ReLU → 2×2 max pool), 16/32/64 filters; global average pooling; dense 32 ReLU; dropout 0.6; dense 1 sigmoid |
| Training | Adam 1e-3, binary cross-entropy, batch 64, 40 epochs, early stopping patience 10 |
| CPU inference | 86 ms per subject, single-threaded |

## Results

**ABIDE I, in-domain** — StratifiedGroupKFold, 5 folds × 10 repeats, 50 fits.
Values are means ± SD across fits.

| | AUROC | AUPRC |
|---|---|---|
| Lightweight CNN | 0.898 ± 0.050 | 0.644 ± 0.141 |
| Random forest on image quality metrics | 0.876 ± 0.068 | 0.602 ± 0.138 |

Chance AUPRC is 0.061 at the cohort prevalence of 6.1%, so the network's lift
is 10.6×. The difference from the random forest is not significant: a paired
subject-level bootstrap gives +0.032 AUROC, 95% CI [−0.024, +0.095].

**MR-ART, zero-shot** — 327 scans, no adaptation, frozen weights.

| | value |
|---|---|
| AUROC | 0.897 (95% CI 0.863–0.929) |
| AUPRC | 0.936 (95% CI 0.912–0.957), chance 0.606 |
| F1 at the inherited 0.5 threshold | 0.717 |
| F1 after recalibration on 10 labelled target subjects | 0.822 ± 0.048 |

Intervals are subject-level cluster bootstraps.

**What bounds these numbers.** Site identity alone, with no image data,
predicts ABIDE quality labels at AUROC 0.848 — most multi-site quality
classifiers are partly recognising scanners. Within site, where that route is
removed by construction, the network retains 0.826 against 0.500 for the
site-only control. Leave-one-site-out AUROC is 0.791 for the network and 0.786
for the metric-based classifier. Agreement between the two human raters is
Cohen's kappa 0.21, which bounds what any model trained on these labels can
achieve.

**Capacity.** Under a protocol matched across architectures, the network falls
0.046 AUROC short of ResNet-50 at 1/912th of its parameters, and exceeds
EfficientNet-B0 and MobileNetV3-Small by 0.049 and 0.068 with bootstrap
intervals excluding zero.

---

## Repository layout

```
code/      analysis scripts, one per reported result (see RESULTS_MAP.md)
cohort/    subject lists and site assignments
models/    trained weights
results/   per-fold predicted probabilities, metrics, figure source data
```

`RESULTS_MAP.md` maps every table and figure in the paper to the script that
produced it and the output file it reads. Every reported value derives from
stored per-fold predictions, so each can be recomputed without retraining.

## Reproducing

Scripts read the prediction files in `results/` and recompute metrics; none
requires a GPU except retraining. Random seed 42 throughout, with repeat *r*
using `random_state = 42 + r`.

Analyses reported at submission used TensorFlow 2.20.0 under Python 3.13.15;
those added in revision used TensorFlow 2.21.0 under Python 3.13.16. Training
used an NVIDIA A100; CPU latency was measured single-threaded on an Intel Xeon
at 2.20 GHz, as a graph-compiled forward pass, median of 30 timed calls after
five warm-ups.

## Data

Neither dataset is redistributed. Both retain their own terms of use.

- **ABIDE I** — http://fcon_1000.projects.nitrc.org/indi/abide/
- **MR-ART** — https://openneuro.org/datasets/ds004173

Quality labels for ABIDE come from the manual ratings distributed with the
dataset, using agreement between two raters. No image quality metric was used
in constructing the labels.

## Citation

```bibtex
@article{kaur2026lightweight,
  title   = {Cross-Dataset Structural Brain MRI Quality Assessment with a
             25,857-Parameter CNN: Capacity Is Not the Limiting Factor},
  author  = {Kaur, Tajinder and Cheema, Sikander Singh and Kaur, Lakhwinder},
  journal = {Magnetic Resonance Imaging},
  year    = {2026},
  doi     = {10.1016/J.MRI.2026.110859}
}
```

## Licence

Code released under the MIT License.
