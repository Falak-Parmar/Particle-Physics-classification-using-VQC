# IT401 — Quantum Machine Learning: Project Progress Summary

## 📌 Project Overview

**Course:** IT401 — Quantum Machine Learning  
**Goal:** Reproduce and extend the paper *"Quantum Machine Learning for Particle Physics using a Variational Quantum Classifier"* by Blance & Spannowsky (JHEP 2021).  
**Task:** Implement a Variational Quantum Classifier (VQC) for signal vs. background event classification in high-energy physics using the HIGGS dataset.

---

## 📄 Reference Paper

| Detail | Info |
|---|---|
| **Title** | QML for Particle Physics using a Variational Quantum Classifier |
| **Authors** | Blance & Spannowsky |
| **Published** | JHEP 2021 |
| **Dataset** | HIGGS dataset |
| **Task** | Binary classification: signal vs. background events |

---

## ✅ What Has Been Done

### 1. Repository Setup
- Git repository initialized
- Organized project structure with dedicated folders for notebooks and utilities

### 2. Classical MLP Baseline (`baseline_mlp.ipynb`)
- Built a classical Multi-Layer Perceptron (MLP) as a comparison baseline
- Used **2 features** from the HIGGS dataset
- Optimizer: **SGD (Stochastic Gradient Descent)**
- Achieved modest **accuracy** and **AUC** scores
- Serves as the benchmark to compare quantum classifier performance against

### 3. Shared Data Utility (`utils/data_utils.py`)
A shared helper module created to ensure consistency across all experiment notebooks:
- HIGGS dataset loading
- Feature selection
- **MinMax scaling to `[0, π]`** (correct preprocessing range for angle encoding)
- **60/20/20 train/validation/test split**

> ⚠️ All notebooks use this shared utility to ensure valid comparisons across experiments.

### 4. Experiment Notebooks (Scaffolded)

#### `01_vqc_2features.ipynb` — Core VQC Reproduction
- Reproduces the paper's core architecture
- **2 features, 2 circuit layers**
- VQC Architecture (as per paper):
  - **Encoding:** Ry angle encoding
  - **Model layers:** Rot gates + CNOT entanglement
  - **Measurement:** Pauli-Z expectation value on qubit 0
  - **Bias:** Trainable bias term
  - **Optimizers:** Standard gradient descent + Quantum Natural Gradient Descent (QNG)

#### `02_feature_scaling.ipynb` — Feature Dimensionality Scaling
- Systematically scales the number of input features
- Tests: **2 → 4 → 6 → 8 features**
- Observes how more input dimensions affect classification performance

#### `03_encoding_strategies.ipynb` — Encoding Strategy Comparison
- Compares different quantum data encoding methods:
  - **Angle encoding**
  - **Amplitude encoding**
  - **Data re-uploading**

### 5. README
A comprehensive `README.md` was written containing:
- Full project structure
- Architecture details
- Experiment table
- Results summary template (to be filled after running experiments)

---

## 🔬 Key Technical Decisions & Learnings

| Topic | Decision |
|---|---|
| **Framework** | PennyLane (quantum ML) |
| **Encoding** | Ry angle encoding (matching the paper) |
| **Preprocessing range** | `[0, π]` for angle encoding |
| **Data split** | 60 / 20 / 20 (train / val / test) |
| **Measurement** | Pauli-Z expectation on qubit 0 |
| **Bias** | Trainable bias included |
| **Optimizers** | Gradient descent + QNG |

---

## ⏳ Pending / On the Horizon

| Item | Status |
|---|---|
| `04_circuit_depth.ipynb` — Circuit depth variations | ⏸ Deferred (awaiting confirmation) |
| `05_dataset_size_scaling.ipynb` — Dataset size scaling | ⏸ Deferred (awaiting confirmation) |
| Run all experiments in order (01 → 02 → 03 → 04 → 05) | 🔲 Not started |
| Populate results summary template with outcomes | 🔲 Not started |
| Compare VQC results vs. classical MLP baseline | 🔲 Not started |

---

## 📁 Project Structure

```
project/
│
├── utils/
│   └── data_utils.py              # Shared data loading & preprocessing
│
├── notebooks/
│   ├── 00_baseline_mlp.ipynb      # Classical MLP baseline
│   ├── 01_vqc_2features.ipynb     # Core VQC reproduction
│   ├── 02_feature_scaling.ipynb   # Feature dimensionality scaling
│   ├── 03_encoding_strategies.ipynb # Encoding strategy comparison
│   ├── 04_circuit_depth.ipynb     # [PENDING] Circuit depth variations
│   └── 05_dataset_size_scaling.ipynb # [PENDING] Dataset size scaling
│
└── README.md                      # Project overview & results template
```

---

## 🗒️ Suggested Experiment Execution Order

```
01_vqc_2features  →  02_feature_scaling  →  03_encoding_strategies  →  04_circuit_depth  →  05_dataset_size_scaling
```

---

*Last updated: April 15, 2026*
