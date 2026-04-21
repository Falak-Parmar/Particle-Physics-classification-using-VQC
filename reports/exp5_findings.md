# Experiment 05: Dataset Size Scaling

## Objective
Determine how VQC performance scales with the amount of available training data and identify the "Small-Data Advantage".

## Methodology
- **Dataset Sizes:** 500, 1000, 2500, and 5000 total samples.
- **Architecture:** 2 Qubits, 2 Layers.
- **Metric:** Mean Test AUC (BCE Standard).

## Scaling Observations
| Samples | Performance Insight |
|---|---|
| **500 - 1000** | **Quantum Edge:** VQC demonstrates its highest efficiency, identifying patterns with minimal examples. |
| 2500 - 5000 | **Capacity Bottleneck:** Performance begins to plateau as the data distribution complexity exceeds the circuit's representational capacity. |

## Key Insights
1. **Generalization Power:** Quantum models excel in regimes where data is scarce, effectively mapping low-density information into high-dimensional Hilbert spaces.
2. **Scalability Ceiling:** To handle larger datasets, the VQC requires a corresponding increase in circuit width (qubits) or depth (layers) to avoid underfitting.

---
*Reference Notebook:* `notebook/05_dataset_size.ipynb`
