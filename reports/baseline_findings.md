# Baseline Evaluation: Classical MLP

## Objective
Establish a high-quality classical benchmark for particle physics classification on the HIGGS dataset to validate the performance of the Variational Quantum Classifier.

## Methodology
- **Model:** Deep Multilayer Perceptron (MLP).
- **Architecture:** (100, 50, 25) Hidden layers.
- **Optimizer:** Adam with Early Stopping.
- **Evaluation:** 5-seed Mean Test AUC (BCE/Log-Loss).

## Classical Benchmarks (N=1000)
| Features | Test AUC (Mean) | Std Dev |
|---|---|---|
| 2 Features | 0.5642 | ± 0.0410 |
| **8 Features** | **0.5831** | **± 0.0519** |

## Key Insights
1. **Information Sensitivity:** The MLP benefits from the additional physics features in the 8-feature set but shows high variance when training data is restricted to 1,000 samples.
2. **Standard of Comparison:** This strong baseline provides the honest classical threshold that the VQC must surpass to demonstrate algorithmic advantage.

---
*Reference Notebook:* `notebook/strong_baseline_mlp.ipynb`
