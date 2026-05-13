# LightPRA: A Lightweight Temporal Convolutional Network for Automatic Physical Rehabilitation Exercise Assessment


[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Dataset](https://img.shields.io/badge/Dataset-KIMORE-green)](https://github.com/lcostantini/Kimore)
[![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle)](https://kaggle.com)

---

## Overview

 The present **LightPRA** architecture(Interpretable Attention extension) we are proposing builds on top of [LightPRA (Sardari et al., 2024)](https://doi.org/10.1016/j.compbiomed.2024.108382) — a lightweight temporal convolutional network for automatic physical rehabilitation assessment — by introducing three architectural novelties that improve both **prediction accuracy** and **clinical interpretability**.

The model is evaluated on the **KIMORE dataset** (78 participants, 5 exercises, scores 0–50 rated by trained physiotherapists) and achieves **10–18% MAD reduction** over the original LightPRA baseline across all five exercises.

---

## Repository Structure

```
Team22_DataAnalyticsProject/
│
├── 📒 lightpra_coordinated.ipynb     # Ex2 (Lateral Tilt) & Ex3 (Trunk Rotation)
├── 📒 lightpra_uniaxial.ipynb        # Ex1 (Lifting Arms), Ex4 (Pelvis Rotation) & Ex5 (Squatting)
│
└── README.md
```

### Notebook Descriptions

#### `lightpra_coordinated.ipynb` — *for Exercise 2 & Exercise 3*
Exercises 2 (Lateral Tilt) and 3 (Trunk Rotation) involve **coordinated trunk-arm movements** where quality depends on precise inter-limb timing. This notebook uses:
- **Learnable sinusoidal temporal attention pooling** — lets the model focus on key exercise phases
- **Cross-anatomical multi-head attention (N2)** — explicitly models coordination between trunk, arms, and legs
- **Consistency regularisation** — penalises variance across a participant's three repetitions

#### `lightpra_uniaxial.ipynb` — *for Exercise 1, 4 & 5*
Exercises 1 (Lifting Arms), 4 (Pelvis Rotation), and 5 (Squatting) are more **cyclical or lower-body dominant**. Learnable temporal weights were found to overfit on these, so this notebook uses:
- **Fixed exponential-decay temporal weights** — emphasises later exercise phases (where fatigue and form degradation are most diagnostic), hardcoded to avoid overfitting
- **SE-Net anatomical gate** — learns per-body-part importance scores
- **Repetition-aware kinematic features (N3)** — captures fatigue-related variability across 3 repetitions

---
## Architecture at a Glance

```
KIMORE Input (25 joints × 4 quaternion components = 100 channels/frame)
          │
          ▼
  Anatomical Reordering → [Trunk | L.Arm | R.Arm | L.Leg | R.Leg]  (80 channels)
          │
          ▼
  Autocorrelation Segmentation → 3 repetitions per recording
          │
          ▼
  Resample to 100 frames → MinMax Norm → Gaussian Smooth → Pad to 104
          │
          ├──────────────────────────────────────────┐
          ▼                                          ▼
  Multi-Scale Subsampling                   N3: Kinematic Features
  (×1, ×2, ×4, ×8)                         (25-dim: velocity,
          │                                 smoothness, skewness,
          ▼                                 inter-rep consistency)
  5× Body-Part TCN Subnetworks
  (kernel=3, dilations=[1,2,4,8,16,32])
          │
          ▼
  [Ex2/3] Cross-Anatomical MHA (N2)
  [Ex1/4/5] Concatenate body parts
          │
          ▼
  3× Global TCN Layers
          │
          ▼
  [Ex2/3] Sinusoidal Temporal Attention Pool (N1)
  [Ex1/4/5] Fixed Exponential-Decay Pool (N1)
          │
          ▼
  SE-Net Anatomical Gate (N1)
          │
          ├──────────── Concatenate ────────────────┐
          │                                          │
          ▼                                          ▼
  TCN Features                             MLP(16→8, GELU+LN+Dropout)
          │                                          │
          └──────────── Concat ─────────────────────┘
                            │
                            ▼
               Regression Head: Dense(24) → Dense(1)
                            │
                            ▼
                     Clinical Score [0, 1]
```

---

## Setup & Usage

### Prerequisites

```bash
pip install keras-tcn openpyxl tensorflow numpy pandas scipy scikit-learn matplotlib
```

### Dataset
KIMORE dataset

### Running

**For Exercise 2 or 3** (coordinated movement pipeline):
```python
# Open lightpra_coordinated.ipynb
EXERCISE = 2   # or 3
```

**For Exercise 1, 4, or 5** (uniaxial/lower-body pipeline):
```python
# Open lightpra_uniaxial.ipynb
EXERCISE = 1   # or 4 or 5
```

Both notebooks are self-contained — run all cells top to bottom. Intermediate arrays are cached as `.npy` files on the first run to speed up subsequent experiments.

---
## 👥 Team

| Name | Roll No. | Contribution |
|---|---|---|
| Aadharsh Ramachandran | 231IT001 | N1 temporal attention (fixed exponential-decay pool), N2 cross-anatomical MHA, body-part TCN sub-network design, repetition segmentation algorithm |
| Jaivant Kosaraju | 231IT028 | N1 learnable sinusoidal attention pool, dataset ingestion & subject-label pairing pipeline, multi-scale input construction (×1/×2/×4/×8) |
| Paluvadi Dinesh Manideep | 231IT047 | N3 repetition-aware feature design (25-dim), feature correlation analysis, ablation visualisations |
| K Akhilesh | 231IT029 | N3 body-part smoothness features, SE part-importance gate, within-subject consistency regularisation loss, training loop & loss function design |

---

## 📚 References

- Sardari et al. (2024). *LightPRA: A Lightweight Temporal Convolutional Network for Automatic Physical Rehabilitation Exercise Assessment.* Computers in Biology and Medicine, 173, 108382.
- Capecci et al. (2019). *The KIMORE Dataset.* IEEE Transactions on Neural Systems and Rehabilitation Engineering, 27(7), 1436–1448.
- Hu et al. (2018). *Squeeze-and-Excitation Networks.* CVPR.
- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
- Bai et al. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271.

---
