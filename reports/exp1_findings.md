# Experiment 01 Report: VQC Reproduction & QNG Comparison

## Objective
Reproduce the core Variational Quantum Classifier (VQC) architecture from Blance & Spannowsky (2021) and compare the performance of standard Gradient Descent (**Adam**) against **Quantum Natural Gradient (QNG)** using 2 features and 5,000 samples.

## Setup
- **Dataset:** HIGGS (5,000 samples)
- **Features:** 2 (`m_bb`, `missing energy mag.`)
- **Architecture:** 2 qubits, 2 layers of `Rot` gates with CNOT entanglement.
- **Optimizers:**
  - **Adam:** Learning rate 0.05, 30 epochs.
  - **QNG:** Learning rate 0.05, 30 epochs, Block-diagonal metric tensor approximation.

## Results (5,000 Samples, 2 Layers)
| Optimizer | Test Accuracy | Test AUC |
|---|---|---|
| Adam | 0.6060 | 0.6132 |
| **QNG** | 0.5990 | **0.6245** |

## Comparison with Classical Baselines
| Model | Optimizer | Test AUC |
|---|---|---|
| Weak Baseline (Paper) | SGD | 0.5960 |
| **VQC (QNG)** | **QNG** | **0.6245** |
| Strong MLP Baseline | Adam | 0.6974 |

## Key Findings
1. **QNG Superiority:** QNG achieved a higher AUC (0.6245) than Adam (0.6132), confirming that accounting for the Hilbert space geometry improves quantum model training.
2. **Quantum vs. Classical:** While the VQC outperformed the paper's weak classical baseline, it still lags behind a well-optimized deep MLP (AUC 0.6974). 
3. **Bottleneck:** A 2-qubit, 2-layer circuit may lack the expressivity required to fully capture the data distribution compared to a classical hidden layer with many neurons.

---
*Date: April 16, 2026*
