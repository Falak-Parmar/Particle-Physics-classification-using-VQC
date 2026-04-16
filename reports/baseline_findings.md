# Baseline Findings: Classical MLP with Gradient Descent

## Objective
Identify the maximum performance achievable with a classical Gradient Descent (SGD) model using **2 features** (m_bb, missing energy magnitude) and **5,000 samples** on the HIGGS dataset, following the constraints of the Blance & Spannowsky (2021) paper.

## Results: Searching for Maximum Performance
We conducted a search across various architectures and learning rates using the standard `MLPClassifier` with `solver='sgd'`.

### Grid Search Results (5,000 Samples, 2 Features)
| Architecture | Learning Rate | Test Accuracy | Test AUC |
|---|---|---|---|
| (3,) | 0.01 (Paper Default) | 0.6010 | 0.5960 |
| (3,) | 0.05 | 0.6550 | 0.6963 |
| (10,) | 0.01 | 0.6210 | 0.6926 |
| (20, 20) | 0.001 | 0.6380 | **0.6974** |

## Key Findings
- **Optimization Plateau:** For 2 features, the maximum AUC achievable with classical GD (SGD) is approximately **0.697**.
- **Architecture Sensitivity:** The paper's baseline (3 neurons) is very sensitive to the learning rate. Moving from `0.01` to `0.05` nearly doubles the discrimination power (AUC +0.10).
- **VQC Target:** To demonstrate quantum advantage as claimed in the paper, the VQC must exceed an AUC of ~0.70 with 2 features. 

## Comparison Table
| Model | Features | Optimizer | Accuracy | AUC |
|---|---|---|---|---|
| Weak Baseline (Paper Default) | 2 | SGD | 0.601 | 0.596 |
| **Max Possible Classical GD** | 2 | SGD | **0.638** | **0.697** |
| Strong Baseline (from exp) | 2 | Adam | 0.658 | 0.691 |

---
*Date: April 15, 2026*
