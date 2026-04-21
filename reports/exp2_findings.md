# Experiment 02: Feature Scaling Analysis

## Objective
Investigate how increasing the input dimensionality (from 2 to 8 qubits) affects the discriminatory power of the VQC on the HIGGS dataset.

## Methodology
- **Features Tested:** 2, 4, and 8 (Selected by correlation ranking).
- **Architecture:** Fixed 2 Layers (Angle Encoding).
- **Optimizer:** Adam (LR 0.05).
- **Evaluation:** BCE Test AUC (N=1000).

## Scaling Trends
| Feature Count | Performance Trend |
|---|---|
| 2 Features | Baseline performance; captures high-level mass correlations. |
| 4 Features | Moderate improvement; integration of low-level momentum features. |
| **8 Features** | **Optimal Performance:** Highest observed AUC; captures complex interactions between jet b-tagging and invariant masses. |

## Key Insights
1. **Information Density:** VQC performance scales positively with feature count, provided the features are ranked by physical relevance.
2. **Qubit Efficiency:** The 8-qubit configuration provides the best balance between representational capacity and simulation stability.

---
*Reference Notebook:* `notebook/02_feature_scaling.ipynb`
