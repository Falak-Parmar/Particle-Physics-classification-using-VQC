# Particle Physics Classification using a Variational Quantum Classifier

A reproduction and extension of **Blance & Spannowsky (JHEP 2021)** — a hybrid quantum-classical machine learning algorithm for event classification in high-energy physics.

> **Course:** IT401 — Quantum Machine Learning  
> **Student:** Falak Parmar (202518053)

---

## Background

Discovering new physics at the LHC requires separating rare signal events from an overwhelming Standard Model background. This project applies a **Variational Quantum Classifier (VQC)** — a hybrid quantum-classical neural network — to this binary classification problem, comparing it against a classical MLP baseline.

The paper demonstrates that a VQC trained with **Quantum Natural Gradient Descent (QNG)** outperforms both a classical NN and a VQC trained with standard gradient descent.

---

## Dataset

We use the **HIGGS dataset** (UCI ML Repository) as a proxy for the paper's particle physics simulation.

- **Original paper's task:** pp → Z′ → tt̄ (signal) vs pp → tt̄ (background) at 14 TeV — *uses a custom simulation, not publicly available*
- **Our dataset:** UCI HIGGS — H → ττ (signal) vs background, 28 features (21 low-level + 7 high-level)
- **Dataset size:** 5,000 samples
- **Split:** 60% train / 20% val / 20% test

### Feature Selection

Since we use a different dataset than the paper, we select features by **correlation analysis** rather than blindly mapping the paper's feature names. The paper uses pT,b1 and E_T^miss; our baseline analysis found the following ranking for the HIGGS dataset:

| Rank | CSV Col | Feature Name | Type |
|------|---------|-------------|------|
| 1 | 26 | m_bb (inv. mass of b-tagged jets) | high-level |
| 2 | 4 | missing energy magnitude | low-level |
| 3 | 28 | m_wwbb | high-level |
| 4 | 1 | lepton pT | low-level |
| 5 | 6 | jet 1 pt | low-level |
| 6 | 13 | jet 2 b-tag | low-level |
| 7 | 27 | m_wbb | high-level |
| 8 | 25 | m_jlv | high-level |

**Default 2-feature set:** columns 26 (m_bb) and 4 (missing energy magnitude)

Place `HIGGS.csv.gz` in the `data/` folder (gitignored due to size).

---

## Architecture

### VQC Pipeline

```
Input (2 features)
    │
    ▼
State Preparation (angle encoding via Ry gates)
    │
    ▼
Model Circuit (L layers of Rot gates + CNOT entanglement)
    │
    ▼
Measurement (⟨σz⟩ on qubit 0)
    │
    ▼
Postprocessing (+ trainable bias b)
    │
    ▼
Classification (threshold at 0)
```

### State Preparation
- Ry(x_i) applied to qubit i, encoding each feature as a rotation angle

### Model Circuit (per layer)
- General Rot(θ, φ, λ) gate on each qubit
- CNOT(0→1) and CNOT(1→0) for entanglement

### Optimization
| Method | Description |
|--------|-------------|
| GD | Standard gradient descent (parameter-shift rule) |
| QNG | Quantum Natural Gradient — uses Fubini-Study metric for faster convergence |

---

## Project Structure

```
.
├── baseline/
│   └── qml_baseline_notebook.ipynb   # Classical MLP (paper-aligned)
├── experiments/
│   ├── 01_vqc_2features.ipynb        # Core VQC reproduction (2 features, 2 layers)
│   ├── 02_feature_scaling.ipynb      # 2→4→6→8 feature experiments
│   ├── 03_encoding_strategies.ipynb  # Angle vs amplitude vs data reuploading
│   ├── 04_circuit_depth.ipynb        # Effect of increasing circuit layers
│   └── 05_dataset_size.ipynb         # 5000→10000+ sample scaling
├── figures/                          # Saved plots and ROC curves
├── notes/                            # Paper notes, derivations
├── utils/
│   └── data_utils.py                 # Shared data loading/preprocessing
├── data/                             # HIGGS.csv.gz (gitignored)
└── README.md
```

---

## Results Summary

| Model | Features | Accuracy | AUC |
|-------|----------|----------|-----|
| Classical MLP (baseline) | col 26, 4 | 0.601 | 0.596 |
| VQC-GD (2 feat, 2 layers) | col 26, 4 | 0.616 | 0.609 |
| VQC-QNG | col 26, 4 | — | — |

*Paper reported (on their own dataset): Classical NN 73.8% AUC, VQC-GD 77.3%, VQC-QNG 79.4%*

---

## Proposed Experiments

| # | Experiment | Question |
|---|-----------|----------|
| 0 | Baseline MLP | Reference classical performance |
| 1 | VQC (2 features, 2 layers) | Can VQC match classical NN? |
| 2a | VQC (4 features) | Does more input information help? |
| 2b | VQC (6 features) | Where does VQC performance plateau? |
| 2c | VQC (8 features) | Does increasing features hurt due to barren plateaus? |
| 3a | Amplitude encoding | How does encoding strategy affect accuracy? |
| 3b | Data reuploading | Can reuploading improve expressibility? |
| 4a | 3-layer VQC | Does depth improve classification? |
| 4b | 4-layer VQC | At what depth does training destabilize? |
| 5a | 10,000 samples | How much does more data help? |
| 5b | 500 samples | Can VQC learn from very limited data? |

---

## Dependencies

```bash
pip install pennylane pennylane-lightning numpy pandas scikit-learn matplotlib
```

---

## References

- Blance & Spannowsky, *Quantum machine learning for particle physics using a variational quantum classifier*, JHEP 02 (2021) 212. [arXiv:2010.07335](https://arxiv.org/abs/2010.07335)
- Baldi et al., *Searching for Exotic Particles in High-Energy Physics with Deep Learning*, Nature Comm. 5 (2014) — HIGGS dataset
