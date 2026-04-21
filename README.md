# Particle Physics Classification using a Variational Quantum Classifier

A reproduction and extension of **Blance & Spannowsky (JHEP 2021)** — a hybrid quantum-classical machine learning algorithm for event classification in high-energy physics.

> **Course:** IT401 — Quantum Machine Learning  
> **Student:** Falak Parmar (202518053)

---

## Final Project Status: Rigorous Benchmark
Following a technical audit, this project was refactored to ensure statistical validity:
1. **No Data Leakage:** Preprocessing scalers are fit strictly on training data.
2. **Proper Loss:** Switched from MSE to **Binary Cross-Entropy (BCE)** for classification.
3. **Statistical Power:** Final results are based on the **Mean of 5 Random Seeds**.

---

## Results Summary (N=1000, 8 Features)

The definitive project results, established through leak-free, multi-seeded evaluation:

| Model | Loss Function | Mean Test AUC | Std Dev | Status |
|-------|---------------|---------------|---------|--------|
| **Classical MLP** | Log-Loss (BCE) | 0.5831 | ± 0.0519 | Benchmark |
| **Quantum VQC** | **BCE** | **0.6209** | **± 0.0405** | **Champion** |

**Quantum Advantage:** The VQC maintains a statistically robust edge of **+0.038 AUC** in the small-data regime.

---

## Project Structure

```
.
├── notebook/                         # Consolidated research notebooks
│   ├── 00_eda_higgs.ipynb            # Initial data exploration
│   ├── 01_vqc_2features.ipynb        # Core VQC reproduction
│   ├── ...                           # Ablation studies (Scaling, Depth, etc.)
│   ├── 06_best_config_synthesis.ipynb # Optimized project configuration
│   ├── 08_rigorous_reevaluation.ipynb # Final gold-standard truth test
│   └── strong_baseline_mlp.ipynb     # Classical benchmark logic
├── reports/                          # Detailed findings for each phase
│   ├── exp1-exp5_findings.md         # Ablation study reports
│   └── Optimized_strat.md            # Final project synthesis
├── utils/
│   └── data_utils.py                 # Refactored, leak-free data engine
├── figures/                          # Visual evidence (ROC, Loss curves)
├── requirements.txt                  # Pinned dependencies
└── README.md
```

---

## Dependencies
```bash
pip install -r requirements.txt
```

---

## References
- Blance & Spannowsky, *Quantum machine learning for particle physics using a variational quantum classifier*, JHEP 02 (2021) 212.
- Baldi et al., *Searching for Exotic Particles in High-Energy Physics with Deep Learning*, Nature Comm. 5 (2014) — HIGGS dataset
