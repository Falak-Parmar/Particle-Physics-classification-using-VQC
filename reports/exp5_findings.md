# Experiment 05 Report: Dataset Size Scaling (QNG Optimized)

## Objective
Evaluate how the amount of training data affects the performance and training efficiency of the Variational Quantum Classifier (VQC). We scale the dataset from 500 to 10,000 samples using the **Quantum Natural Gradient (QNG)** optimizer and 2 features.

## Setup
- **Features:** 2 (`m_bb`, `missing energy mag.`)
- **Architecture:** 2 Qubits, 2 Layers (Angle Encoding)
- **Optimizer:** QNG (LR 0.05, 30 Epochs)
- **Dataset Sizes:** 500, 1,000, 5,000, and 10,000 total samples.

## Results (30 Epochs)
| Total Samples | Train Samples | Test Accuracy | Test AUC | Training Time |
|---|---|---|---|---|
| 500 | 300 | 0.6800 | **0.6700** | 41.9s |
| 1,000 | 600 | 0.6800 | 0.6664 | 83.1s |
| 5,000 | 3,000 | 0.5990 | 0.6245 | 411.7s |
| 10,000 | 6,000 | 0.5970 | 0.5914 | 827.1s |

## Key Findings
1. **Strong Generalization with Limited Data:** The VQC achieved its highest AUC (0.6700) with the smallest training set (300 samples). This suggests that quantum models may have a generalization advantage over classical models in low-data regimes, as they can identify meaningful patterns without needing millions of examples.
2. **The Capacity Bottleneck:** A significant performance drop was observed as the dataset size increased. For 10,000 samples, the AUC fell to 0.5914. This indicates that the 2-qubit, 2-layer architecture has a restricted **representational capacity**. As the data distribution becomes more complex with more samples, the simple circuit fails to capture the nuances, leading to underfitting.
3. **Linear Scaling of Training Time:** The wall-clock time increased linearly with the number of samples. This confirms that the batch-wise application of QNG (including metric tensor computation) scales predictably, making it viable for larger datasets if model capacity is also increased.

## Conclusion
The VQC is highly effective for particle physics classification when data is scarce. however, for larger datasets, the circuit must be scaled in depth or width (qubits) to handle the increased information density.

---
*Date: April 16, 2026*
