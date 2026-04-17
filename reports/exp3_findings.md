# Experiment 03 Report: Encoding Strategy Comparison (QNG Optimized)

## Objective
Identify the most effective quantum data encoding strategy for particle physics classification using the HIGGS dataset (5,000 samples, 2 features). We compare three core methods using the **Quantum Natural Gradient (QNG)** optimizer to ensure a fair and optimized comparison.

## Setup
- **Features:** 2 (`m_bb`, `missing energy mag.`)
- **Optimizer:** QNG (LR 0.05, 30 Epochs)
- **Architecture:** 2 Layers (CNOT entanglement)

## Results (5,000 Samples, 30 Epochs)
| Strategy | Test Accuracy | Test AUC |
|---|---|---|
| **Angle Encoding** (Paper Default) | 0.5990 | **0.6245** |
| Data Reuploading | 0.5990 | 0.6097 |
| Amplitude Encoding | 0.5530 | 0.5898 |

## Key Findings
1. **Angle Encoding Lead:** Angle encoding remains the most effective method for this shallow VQC setup. It maintains the highest discriminatory power (AUC 0.6245) compared to more complex strategies.
2. **Reuploading Performance:** Data reuploading (re-encoding features at each layer) did not outperform the baseline angle encoding. In this shallow (2-layer) circuit, the extra complexity of the energy landscape may not yet be translating into better separation.
3. **Amplitude Encoding Issues:** Normalizing inputs for amplitude encoding consistently results in lower performance. This is likely because the absolute magnitudes of certain physics features (like missing energy) are critical for classification and are lost during the normalization process.

## Conclusion
Angle encoding is established as the best-performing strategy for low-dimensional VQC classification in this project. All subsequent experiments (Scaling, Depth, etc.) will continue to use Angle encoding as the standard state preparation method.

---
*Date: April 16, 2026*
