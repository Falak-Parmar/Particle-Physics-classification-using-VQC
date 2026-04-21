# Experiment 04: Circuit Depth Analysis

## Objective
Investigate the relationship between the number of variational layers (1 to 4) and the classification performance of the VQC.

## Methodology
- **Features:** 2 (`m_bb`, `missing energy mag.`)
- **Layers Tested:** 1, 2, 3, and 4.
- **Entanglement:** Circular CNOT chain.
- **Training:** 30 Epochs (BCE Loss).

## Depth Trends
| Depth | Observation |
|---|---|
| 1 Layer | Insufficient expressivity; limited discriminatory power. |
| 2 Layers | **Efficiency Peak:** Optimal balance of learning capacity and training stability. |
| 3-4 Layers | **Diminishing Returns:** Increased depth adds parameter complexity without proportional gains in AUC for this feature set. |

## Key Insights
1. **Optimal Capacity:** For low-dimensional data (2 features), a 2-to-3 layer circuit provides the sufficient "space folding" required for classification.
2. **Training Stability:** Deeper circuits require more careful initialization and optimization to avoid performance plateaus.

---
*Reference Notebook:* `notebook/04_circuit_depth.ipynb`
