# Experiment 01: VQC Core Reproduction

## Objective
Reproduce the foundational Variational Quantum Classifier (VQC) architecture (2 features, 2 layers) and compare the performance of standard Gradient Descent against the **Quantum Natural Gradient (QNG)**.

## Methodology
- **Features:** 2 (`m_bb`, `missing energy mag.`)
- **Architecture:** 2 Qubits, 2 Layers (Angle Encoding)
- **Training:** 30 Epochs, Adam vs. QNG.
- **Data Pipeline:** Standardized BCE classification (N=1000).

## Comparative Trends
| Optimizer | Observed Trend |
|---|---|
| Adam | Steady convergence, standard parameter-space descent. |
| **QNG** | **Superior Convergence:** Faster reduction in loss by accounting for the Fubini-Study metric of the Hilbert space. |

## Key Insights
1. **Geometric Advantage:** QNG consistently outperforms standard gradient descent in quantum circuits by navigating the state space more effectively.
2. **Initial Benchmarking:** A 2-qubit VQC successfully captures basic physics correlations but requires higher dimensionality to compete with deep classical models.

---
*Reference Notebook:* `notebook/01_vqc_2features.ipynb`
