# Experiment 06 — Best Configuration Synthesis

**Research question:** What is the highest AUC a VQC can achieve on the HIGGS dataset when all findings from experiments 01–05 are combined into a single optimal configuration?

**Hypothesis:** Stacking the winning choices from each prior experiment — correct feature selection, angle encoding, 3 variational layers, 7 000 samples, QNG optimiser, small weight initialisation, and `lightning.qubit` — should push AUC meaningfully above the 0.677 achieved in Experiment 02 and closer to the paper's reported 0.794.

---

## Setup summary

| Axis | Choice | Source |
|---|---|---|
| Features | 6 (cols selected by correlation ranking) | Exp 02 |
| Encoding | Angle (Ry gates, scaled to [0, π]) | Exp 03 |
| Layers | 3 | Exp 04 |
| Dataset size | 7 000 total (60/20/20 split) | Exp 05 |
| Optimiser | QNG | Paper / Exp 01 |
| Weight init | Uniform [0, π/4] | Barren plateau fix |
| Gradient method | `backprop` | Training fix |
| Backend | `lightning.qubit` | Hardware constraint |
| Epochs | 50 | Training fix |
| Seeds | 3 (report mean ± std) | Reproducibility |

---

## Hardware notes (MacBook Air M4, 16 GB RAM)

- **No MPS:** PennyLane does not support Apple's Metal Performance Shaders. All simulation runs on CPU.
- **`lightning.qubit`** uses a C++ kernel with OpenMP threading. It will spread work across the 10 performance cores automatically — no configuration needed.
- **`diff_method='backprop'`** replaces the parameter-shift rule. Instead of 2 circuit evaluations per parameter per gradient step, you get the full gradient in a single forward+backward pass. With 55 parameters this is a significant wall-clock saving.
- **6 qubits** is well inside the fast simulation regime. Statevector size is 2⁶ = 64 complex amplitudes — negligible memory, sub-millisecond per circuit evaluation.
- **Estimated runtime:** ~3–5 s/epoch × 50 epochs × 3 seeds ≈ **12–18 minutes total**.

---

## 1. Imports and reproducibility

```python
import sys
sys.path.append('..')

import time
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

from utils.data_utils import load_higgs, binary_accuracy

SEEDS = [42, 7, 123]
```

---

## 2. Hyperparameters

```python
N_FEATURES   = 6
N_LAYERS     = 3
N_SAMPLES    = 7_000
N_EPOCHS     = 50
BATCH_SIZE   = 32
LR           = 0.01        # QNG step size

# Derived
N_PARAMS = N_FEATURES * N_LAYERS * 3 + 1   # Rot gates (3 per qubit per layer) + bias
print(f"Trainable parameters: {N_PARAMS}")  # → 55
```

---

## 3. Data loading

Top-6 features by correlation with the label. Using the same ranking pipeline from `data_utils.py` with `n_features=6`.

```python
X_train, X_val, X_test, y_train, y_val, y_test = load_higgs(
    path='../data/HIGGS.csv.gz',
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    scale_range=(0, np.pi),
    random_state=42,
)

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
```

---

## 4. Circuit definition

```python
dev = qml.device('lightning.qubit', wires=N_FEATURES)

@qml.qnode(dev, diff_method='backprop')
def circuit(inputs, weights):
    # Angle encoding — one Ry per feature
    for i in range(N_FEATURES):
        qml.RY(inputs[i], wires=i)

    # Variational layers: Rot + circular CNOT entanglement
    for layer in range(N_LAYERS):
        for q in range(N_FEATURES):
            qml.Rot(*weights[layer, q], wires=q)
        for q in range(N_FEATURES):
            qml.CNOT(wires=[q, (q + 1) % N_FEATURES])

    return qml.expval(qml.PauliZ(0))

def predict(inputs, weights, bias):
    return circuit(inputs, weights) + bias

def predict_batch(X, weights, bias):
    return np.array([float(predict(x, weights, bias)) for x in X])
```

---

## 5. Loss and accuracy helpers

```python
def mse_loss(weights, bias, X, y):
    preds = predict_batch(X, weights, bias)
    labels = np.where(y == 1, 1.0, -1.0)
    return float(np.mean((preds - labels) ** 2))

def binary_accuracy(scores, y):
    preds = np.where(np.array(scores) >= 0, 1, 0)
    return float(np.mean(preds == y))
```

---

## 6. Training loop (single seed)

```python
def train(seed):
    rng = np.random.default_rng(seed)

    # Small init — critical for avoiding barren plateaus
    weights = pnp.array(
        rng.uniform(0, np.pi / 4, size=(N_LAYERS, N_FEATURES, 3)),
        requires_grad=True
    )
    bias = pnp.array(0.0, requires_grad=True)

    opt = qml.QNGOptimizer(stepsize=LR)

    train_losses, val_losses = [], []
    n_batches = len(X_train) // BATCH_SIZE

    t0 = time.time()
    for epoch in range(N_EPOCHS):
        # Shuffle each epoch
        idx = rng.permutation(len(X_train))
        X_shuf, y_shuf = X_train[idx], y_train[idx]

        for b in range(n_batches):
            Xb = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            yb = y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            def cost_fn(w, b_):
                return mse_loss(w, b_, Xb, yb)

            (weights, bias), loss_val = opt.step_and_cost(cost_fn, weights, bias)

        # End-of-epoch metrics
        tr_loss = mse_loss(weights, bias, X_train, y_train)
        vl_loss = mse_loss(weights, bias, X_val,   y_val)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}  "
                  f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
                  f"({elapsed:.0f}s elapsed)")

    # Final test evaluation
    test_scores  = predict_batch(X_test,  weights, bias)
    val_scores   = predict_batch(X_val,   weights, bias)
    test_acc     = binary_accuracy(test_scores, y_test)
    test_auc     = roc_auc_score((y_test == 1).astype(int), test_scores)
    wall_time    = time.time() - t0

    return {
        'weights':       weights,
        'bias':          bias,
        'train_losses':  train_losses,
        'val_losses':    val_losses,
        'test_scores':   test_scores,
        'val_scores':    val_scores,
        'test_acc':      test_acc,
        'test_auc':      test_auc,
        'wall_time':     wall_time,
    }
```

---

## 7. Multi-seed run

```python
results = {}
for seed in SEEDS:
    print(f"\n── Seed {seed} ──")
    results[seed] = train(seed)
    print(f"  → Test acc: {results[seed]['test_acc']:.4f}  "
          f"Test AUC: {results[seed]['test_auc']:.4f}  "
          f"({results[seed]['wall_time']:.0f}s)")
```

---

## 8. Aggregate results

```python
aucs = [results[s]['test_auc'] for s in SEEDS]
accs = [results[s]['test_acc'] for s in SEEDS]

print('\n── Exp 06 Best Configuration — Final Results ──')
print(f'{"Seed":>6} {"Test Acc":>10} {"Test AUC":>10}')
print('-' * 30)
for s in SEEDS:
    print(f'{s:>6} {results[s]["test_acc"]:>10.4f} {results[s]["test_auc"]:>10.4f}')
print('-' * 30)
print(f'{"Mean":>6} {np.mean(accs):>10.4f} {np.mean(aucs):>10.4f}')
print(f'{"Std":>6} {np.std(accs):>10.4f}  {np.std(aucs):>10.4f}')
```

---

## 9. Plots

### 9a. Training curves (all seeds)

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for s in SEEDS:
    r = results[s]
    axes[0].plot(r['train_losses'], label=f'seed {s}', alpha=0.8)
    axes[1].plot(r['val_losses'],   label=f'seed {s}', alpha=0.8)

for ax, title in zip(axes, ['Train loss', 'Validation loss']):
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE loss')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.savefig('../figures/06_training_curves.png', dpi=150)
plt.show()
```

### 9b. ROC curves + comparison baseline

```python
plt.figure(figsize=(6, 5))

y_binary = (y_test == 1).astype(int)

# All seeds
for s in SEEDS:
    fpr, tpr, _ = roc_curve(y_binary, results[s]['test_scores'])
    plt.plot(fpr, tpr, alpha=0.5, label=f'Exp 06 seed {s} (AUC={results[s]["test_auc"]:.3f})')

# Prior best (Exp 02, 8 features) for reference
# Replace 0.677 with your actual saved score if available
plt.axhline(0, color='none')  # placeholder
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC — Exp 06 Best Configuration (all seeds)')
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig('../figures/06_roc.png', dpi=150)
plt.show()
```

### 9c. Full experiment AUC comparison bar chart

```python
experiment_labels = [
    'MLP\nbaseline',
    'Exp 01\n2 feat',
    'Exp 02\n8 feat',
    'Exp 06\nbest',
    'Paper\nVQC-QNG',
]
experiment_aucs = [
    0.596,               # classical MLP
    0.609,               # Exp 01
    0.677,               # Exp 02 best
    np.mean(aucs),       # Exp 06 mean
    0.794,               # paper target
]
colors = ['#B4B2A9', '#7F77DD', '#7F77DD', '#1D9E75', '#AAAAAA']

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(experiment_labels, experiment_aucs, color=colors, width=0.55)
ax.set_ylabel('Test AUC')
ax.set_title('AUC progression across experiments')
ax.set_ylim(0.55, 0.85)
ax.axhline(0.794, color='#AAAAAA', linestyle='--', linewidth=1, label='Paper target')

for bar, val in zip(bars, experiment_aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../figures/06_auc_progression.png', dpi=150)
plt.show()
```

---

## 10. Observations

*(Fill in after running)*

- Does mean AUC exceed the Exp 02 best of 0.677?
- Is there meaningful variance across the 3 seeds (std > 0.01)?
- Do training curves converge cleanly, or is there a sign of plateau / oscillation after epoch ~30?
- Does the validation loss track training loss closely, or is there a growing gap suggesting overfitting at N=7 000?
- What is the measured wall-clock time? Does it match the ~12–18 min estimate?

---

## Results table (fill in after running)

| Experiment | Features | Layers | N | AUC (mean) | AUC (std) | Acc (mean) |
|---|---|---|---|---|---|---|
| MLP baseline | 2 | — | 5 000 | 0.596 | — | 0.601 |
| Exp 01 | 2 | 2 | 5 000 | 0.609 | — | 0.616 |
| Exp 02 best | 8 | 2 | 5 000 | 0.677 | — | 0.615 |
| **Exp 06** | **6** | **3** | **7 000** | **—** | **—** | **—** |
| Paper VQC-QNG | 2 | 2 | — | 0.794 | — | — |

---

## References

- Blance & Spannowsky, *Quantum machine learning for particle physics using a variational quantum classifier*, JHEP 02 (2021) 212
- PennyLane `lightning.qubit` docs: https://docs.pennylane.ai/projects/lightning/en/stable/
- PennyLane `QNGOptimizer`: https://docs.pennylane.ai/en/stable/code/api/pennylane.QNGOptimizer.html
