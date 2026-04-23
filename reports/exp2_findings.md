# Experiment 02: Feature Scaling Analysis

## Objective
Investigate how increasing the input dimensionality (from 2 to 8 qubits) affects the discriminatory power of the VQC on the HIGGS dataset.

## Methodology
- **Features Tested:** 2, 4, and 8 (Selected by correlation ranking).
- **Architecture:** 2 Layers (Angle Encoding).
- **Optimizer:** Adam.

## Scaling Trends (Observed Reproduction)
| Feature Count | Test Accuracy | Test AUC |
|---|---|---|
| 2 Features | 0.616 | 0.609 |
| 4 Features | 0.608 | 0.613 |
| 6 Features | 0.616 | 0.673 |
| **8 Features** | **0.615** | **0.677** |

## Key Insights
1. **Dominant Gain:** Scaling 2 → 8 features resulted in a +6.8% AUC improvement, the largest single gain in the ablation study.
2. **Information density:** Performance scales positively with feature count, confirming that the VQC can successfully integrate low-level momentum features with high-level mass features.

---
*Reference Notebook:* `notebook/02_feature_scaling.ipynb`
