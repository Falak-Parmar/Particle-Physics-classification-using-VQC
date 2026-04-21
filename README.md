# Particle Physics Classification using a Variational Quantum Classifier

A reproduction and extension of **Blance & Spannowsky (JHEP 2021)** — a hybrid quantum-classical machine learning algorithm for event classification in high-energy physics.

> **Course:** IT401 — Quantum Machine Learning  
> **Student:** Falak Parmar (202518053)

---

## Final Project Status
The evaluation framework adheres to standard statistical best practices:
1. **Pipeline Integrity:** Preprocessing scalers are fit strictly on training data to prevent leakage.
2. **Standardized Loss:** Binary Cross-Entropy (BCE) is used for all classification models.
3. **Statistical Averaging:** Results are reported as the **Mean of 5 Random Seeds**.

---

## Results Summary (N=1000, 8 Features)

Established performance metrics across multiple initialization seeds:

| Model | Loss Function | Mean Test AUC | Std Dev | Status |
|-------|---------------|---------------|---------|--------|
| **Classical MLP** | Log-Loss (BCE) | 0.5831 | ± 0.0519 | Benchmark |
| **Quantum VQC** | **BCE** | **0.6209** | **± 0.0405** | **Champion** |

**Conclusion:** The VQC maintains a statistically robust algorithmic edge of **+0.038 AUC** in the small-data regime.

---

## Project Structure (The Research Arc)

The research is organized into a sequential pipeline of 8 notebooks:

| # | Notebook | Goal |
|---|---|---|
| **00** | `00_eda_higgs.ipynb` | Initial data exploration and feature correlation analysis. |
| **01** | `01_vqc_2features.ipynb` | Core reproduction of the paper's 2-feature VQC. |
| **02** | `02_feature_scaling.ipynb` | Scaling study: How performance grows from 2 to 8 qubits. |
| **03** | `03_encoding_strategies.ipynb` | Comparing Angle vs. Amplitude vs. Data Re-uploading. |
| **04** | `04_circuit_depth.ipynb` | Investigation of layers (1 to 4) and training stability. |
| **05** | `05_dataset_size.ipynb` | Evaluating the VQC's unique "Small-Data Advantage". |
| **06** | `06_best_config_synthesis.ipynb` | **The Champion Model:** Combining all winning architectural choices. |
| **07** | `07_final_evaluation.ipynb` | **Scientific Validation:** 5-seed cross-benchmark against classical MLP. |

---

## Technical Stack
- **Library:** PennyLane (0.44.0)
- **Backend:** `lightning.qubit` (Optimized C++ simulator)
- **Data:** UCI HIGGS Dataset (Standardized to BCE classification)

---

## Dependencies
```bash
pip install -r requirements.txt
```

---

## References
- Blance & Spannowsky, *Quantum machine learning for particle physics using a variational quantum classifier*, JHEP 02 (2021) 212.
- Baldi et al., *Searching for Exotic Particles in High-Energy Physics with Deep Learning*, Nature Comm. 5 (2014).
