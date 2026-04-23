# Particle Physics Classification using a Variational Quantum Classifier

A reproduction and extension of **Blance & Spannowsky (JHEP 2021)** — a hybrid quantum-classical machine learning algorithm for event classification in high-energy physics.

> **Course:** IT401 — Quantum Machine Learning  
> **Student:** Falak Parmar (202518053)

---

## Performance Summary

Established performance metrics from the research arc, comparing the classical baseline against our optimized quantum configurations.

### 1. Research Arc Results (Ablation Phase)
| Phase | Model | Features | Samples | Test AUC |
|---|---|---|---|---|
| Baseline | Classical MLP | 2 | 5,000 | 0.596 |
| Exp 01 | VQC Reproduction | 2 | 5,000 | 0.609 |
| Exp 02 | **VQC Feature Scaling** | **8** | **5,000** | **0.677** |
| Exp 03 | Encoding (Re-uploading) | 2 | 5,000 | 0.600 |
| Exp 05 | VQC Small-Data | 2 | 500 | 0.670 |

### 2. Final Statistical Validation (Rigorous Phase)
Validated using a leak-free pipeline and 5-seed statistical averaging (N=1000, 8 features):

| Model | Loss Function | Mean Test AUC | Std Dev |
|---|---|---|---|
| Classical MLP | Log-Loss (BCE) | 0.5831 | ± 0.0519 |
| **Quantum VQC** | **BCE** | **0.6209** | **± 0.0405** |

---

## Project Structure

The research is organized into a sequential pipeline of 8 notebooks:

| # | Notebook | Goal |
|---|---|---|
| **00** | `00_eda_higgs.ipynb` | Initial data exploration and feature correlation analysis. |
| **01** | `01_vqc_2features.ipynb` | Core reproduction of the paper's 2-feature VQC. |
| **02** | `02_feature_scaling.ipynb` | Scaling study: Performance growth from 2 to 8 qubits. |
| **03** | `03_encoding_strategies.ipynb` | Comparing Angle vs. Amplitude vs. Data Re-uploading. |
| **04** | `04_circuit_depth.ipynb` | Investigation of layers (1 to 4) and training stability. |
| **05** | `05_dataset_size.ipynb` | Evaluating the VQC's unique "Small-Data Advantage". |
| **06** | `06_best_config_synthesis.ipynb` | **The Champion Model:** Combining all winning architectural choices. |
| **07** | `07_final_evaluation.ipynb` | **Scientific Validation:** 5-seed statistical cross-benchmark. |

---

## Technical Stack
- **Library:** PennyLane (0.44.0)
- **Backend:** `lightning.qubit` (Optimized C++ simulator)
- **Data:** UCI HIGGS Dataset

---

## Dependencies
```bash
pip install -r requirements.txt
```

---

## References
- Blance & Spannowsky, *Quantum machine learning for particle physics using a variational quantum classifier*, JHEP 02 (2021) 212.
- Baldi et al., *Searching for Exotic Particles in High-Energy Physics with Deep Learning*, Nature Comm. 5 (2014).
