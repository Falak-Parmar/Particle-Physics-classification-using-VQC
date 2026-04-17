# Experiment 04 Report: Circuit Depth Variations (QNG Optimized)

## Objective
Investigate how the number of variational layers affects VQC classification performance for particle physics data. We vary the depth from 1 to 4 layers using the **Quantum Natural Gradient (QNG)** optimizer and 2 features.

## Setup
- **Features:** 2 (`m_bb`, `missing energy mag.`)
- **Encoding:** Angle Encoding
- **Optimizer:** QNG (LR 0.05, 30 Epochs)
- **Depths Tested:** 1, 2, 3, and 4 layers.

## Results (5,000 Samples, 30 Epochs)
| Layers | Parameters | Test Accuracy | Test AUC |
|---|---|---|---|
| 1 | 7 | 0.5590 | 0.5310 |
| 2 | 13 | 0.5990 | 0.6245 |
| 3 | 19 | 0.6070 | 0.6236 |
| 4 | 25 | 0.5880 | **0.6322** |

## Key Findings
1. **Significant Boost from 1 to 2 Layers:** Moving from a single layer to two layers nearly doubled the discrimination power (AUC jump from 0.531 to 0.624). This highlights the necessity of entanglement and multi-layered rotation gates for capturing data correlations.
2. **AUC Plateau:** Performance essentially plateaus at 2 layers. While 4 layers achieved the highest AUC (0.6322), the improvement is marginal given the nearly double number of parameters compared to 2 layers.
3. **Parameter Efficiency:** The 2-layer architecture remains the most efficient "sweet spot" for this 2-feature problem, balancing performance and computational overhead.
4. **Classical Gap:** Even at 4 layers, the VQC (AUC 0.632) still hasn't overtaken the **Strong Classical Baseline (AUC 0.697)** using just 2 features. This suggests that depth alone is not the primary driver for quantum advantage in this low-dimensional feature space.

## Conclusion
Increased circuit depth improves the model's expressivity up to a point, but marginal gains decrease significantly after 2 layers for 2-feature classification.

---
*Date: April 16, 2026*
