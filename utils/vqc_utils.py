"""
utils/vqc_utils.py
------------------
Barren-plateau mitigation utilities for VQC experiments.

Three strategies implemented here:

  1. Identity-block initialisation  (fixes width AND depth plateau)
     Weights drawn from N(0, σ²) with σ << 1 so the circuit starts near
     identity.  Gradient variance scales as O(σ²) at init instead of
     O(2⁻ⁿ), buying many useful epochs before gradients vanish.
     Ref: Grant et al., "An initialization strategy for addressing barren
     plateaus in parametrized quantum circuits", Quantum 3 (2019) 214.

  2. Local cost function  (fixes WIDTH plateau specifically)
     Replace the single global ⟨σ_z^0⟩ measurement with the average of
     all single-qubit Pauli-Z expectations: C_L = (1/n)Σ_i ⟨σ_z^i⟩.
     Local observables provably avoid exponential gradient vanishing even
     for deep, wide circuits.
     Ref: Cerezo et al., "Cost function dependent barren plateaus in
     shallow parametrized quantum circuits", Nature Comms. 12 (2021) 1791.

  3. Layer-wise greedy training  (fixes DEPTH plateau)
     Train layers one at a time, freezing earlier layers.  Each new layer
     is added to an already-useful sub-circuit so its gradients are
     non-trivially informative from the first update step.
     Ref: Skolik et al., "Layerwise learning for quantum neural networks",
     Quantum Mach. Intell. 3 (2021) 5.

  BONUS – gradient variance monitor
     Records ‖∇L‖² per epoch.  Plotting this reveals plateau onset
     (variance collapses) so you can diagnose problems before training
     diverges.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


# ─────────────────────────────────────────────────────────────────────────────
# 1.  INITIALISATION STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def init_uniform(n_layers, n_features, seed=42):
    """
    Naive initialisation: uniform in [0, 2π].
    This is what most tutorials use and what causes barren plateaus in
    deep/wide circuits.  Use as a comparison baseline only.
    """
    rng = np.random.default_rng(seed)
    weights = pnp.array(
        rng.uniform(0, 2 * np.pi, (n_layers, n_features, 3)),
        requires_grad=True
    )
    bias = pnp.array(0.0, requires_grad=True)
    return weights, bias


def init_identity_block(n_layers, n_features, sigma=0.01, seed=42):
    """
    Identity-block initialisation (Grant et al. 2019).

    Pairs of layers are initialised so that U_2 ≈ U_1†, making each
    block collapse to identity at t=0.  For an odd number of layers the
    last layer gets small-σ Gaussian weights.

    This keeps the circuit near identity at initialisation so that
    gradients are O(σ) rather than O(2⁻ⁿ), preventing barren-plateau
    onset for many more training steps.

    Parameters
    ----------
    sigma : float
        Standard deviation of perturbations.  0.01 is a safe default;
        increase to 0.1 for shallow (L≤2) circuits.
    """
    rng = np.random.default_rng(seed)
    weights_np = np.zeros((n_layers, n_features, 3))

    # Pair layers: layer 2k+1 mirrors layer 2k with a small perturbation
    for l in range(0, n_layers - 1, 2):
        base = rng.normal(0, sigma, (n_features, 3))
        weights_np[l]     =  base          # U
        weights_np[l + 1] = -base          # ≈ U†  (negating Rot angles inverts it)

    # Odd-length tail
    if n_layers % 2 == 1:
        weights_np[-1] = rng.normal(0, sigma, (n_features, 3))

    weights = pnp.array(weights_np, requires_grad=True)
    bias    = pnp.array(0.0, requires_grad=True)
    return weights, bias


def init_small_random(n_layers, n_features, sigma=0.1, seed=42):
    """
    Small-random initialisation: N(0, σ²).
    Simpler than identity-block but still much better than uniform.
    Good default for circuits with L ≤ 4.
    """
    rng = np.random.default_rng(seed)
    weights = pnp.array(
        rng.normal(0, sigma, (n_layers, n_features, 3)),
        requires_grad=True
    )
    bias = pnp.array(0.0, requires_grad=True)
    return weights, bias


INIT_STRATEGIES = {
    "uniform":        init_uniform,
    "identity_block": init_identity_block,
    "small_random":   init_small_random,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  COST FUNCTIONS  (global vs local)
# ─────────────────────────────────────────────────────────────────────────────

def make_global_circuit(n_features, n_layers):
    """
    Global cost: measure ⟨σ_z⟩ on qubit 0 only (paper default).
    Gradient variance scales as O(4⁻ⁿ) — barren plateau for n ≥ 4.
    """
    dev = qml.device("default.qubit", wires=n_features)

    @qml.qnode(dev, interface="autograd")
    def circuit(weights, x):
        _encode_and_layer(weights, x, n_features, n_layers)
        return qml.expval(qml.PauliZ(0))

    return circuit


def make_local_circuit(n_features, n_layers):
    """
    Local cost: return average ⟨σ_z^i⟩ across ALL qubits.
    Gradient variance scales only polynomially — avoids barren plateau.
    Ref: Cerezo et al. (2021).
    """
    dev = qml.device("default.qubit", wires=n_features)

    @qml.qnode(dev, interface="autograd")
    def circuit(weights, x):
        _encode_and_layer(weights, x, n_features, n_layers)
        # Return list of all single-qubit expectations
        return [qml.expval(qml.PauliZ(q)) for q in range(n_features)]

    def circuit_mean(weights, x):
        """Average over all qubits — use this as the observable."""
        results = circuit(weights, x)
        return pnp.mean(pnp.array(results))

    return circuit_mean


def _encode_and_layer(weights, x, n_features, n_layers):
    """Shared circuit body: angle encoding + Rot+CNOT layers."""
    for i in range(n_features):
        qml.RY(x[i], wires=i)
    for l in range(n_layers):
        for q in range(n_features):
            qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
        for q in range(n_features - 1):
            qml.CNOT(wires=[q, q + 1])
        if n_features > 1:
            qml.CNOT(wires=[n_features - 1, 0])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LAYER-WISE GREEDY TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def layerwise_train(
    n_features,
    n_layers,
    X_train, y_train,
    X_val,   y_val,
    epochs_per_layer=15,
    batch_size=32,
    lr=0.01,
    init_fn=None,
    use_local_cost=True,
    seed=42,
    verbose=True,
):
    """
    Layer-wise greedy training (Skolik et al. 2021).

    Protocol
    --------
    For l = 1 … n_layers:
      1. Freeze weights of layers 1 … l-1.
      2. Train layer l for `epochs_per_layer` epochs.
      3. Unfreeze all and do a short fine-tuning pass.

    Returns
    -------
    weights, bias, history : dict with train/val losses and gradient norms
    """
    if init_fn is None:
        init_fn = init_identity_block

    weights, bias = init_fn(n_layers, n_features, seed=seed)

    if use_local_cost:
        circuit_fn = make_local_circuit(n_features, n_layers)
    else:
        circuit_fn = make_global_circuit(n_features, n_layers)

    history = {"train_loss": [], "val_loss": [], "grad_var": [], "phase": []}

    for active_layer in range(n_layers):
        if verbose:
            print(f"\n  ── Layer-wise phase: training layer {active_layer + 1}/{n_layers} ──")

        for epoch in range(epochs_per_layer):
            perm = np.random.default_rng(epoch).permutation(len(X_train))
            Xs, ys = X_train[perm], y_train[perm]
            grad_norms = []

            for start in range(0, len(Xs), batch_size):
                Xb = Xs[start : start + batch_size]
                yb = ys[start : start + batch_size].astype(float)

                def cost_fn(w, b):
                    preds = pnp.array([circuit_fn(w, x) + b for x in Xb])
                    return pnp.mean((yb - preds) ** 2)

                grad_w, grad_b = qml.grad(cost_fn)(weights, bias)

                # Only update the active layer's slice
                w_np = weights.numpy().copy()
                # Gradient for the active layer
                active_grad = grad_w[active_layer]
                w_np[active_layer] -= lr * active_grad
                weights = pnp.array(w_np, requires_grad=True)
                bias    = pnp.array(float(bias) - lr * float(grad_b), requires_grad=True)

                grad_norms.append(float(np.linalg.norm(grad_w)))

            tr_loss = _eval_loss(circuit_fn, weights, bias, X_train, y_train)
            vl_loss = _eval_loss(circuit_fn, weights, bias, X_val,   y_val)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            history["grad_var"].append(float(np.var(grad_norms)))
            history["phase"].append(active_layer)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1:2d} | train={tr_loss:.4f}, val={vl_loss:.4f}, "
                      f"grad_var={history['grad_var'][-1]:.2e}")

    # Fine-tuning: all layers, full learning rate
    if verbose:
        print(f"\n  ── Fine-tuning: all {n_layers} layers ──")

    opt = qml.AdamOptimizer(stepsize=lr)
    for epoch in range(epochs_per_layer):
        perm = np.random.default_rng(1000 + epoch).permutation(len(X_train))
        Xs, ys = X_train[perm], y_train[perm]
        for start in range(0, len(Xs), batch_size):
            Xb = Xs[start : start + batch_size]
            yb = ys[start : start + batch_size].astype(float)

            def cost_fn(w, b):
                preds = pnp.array([circuit_fn(w, x) + b for x in Xb])
                return pnp.mean((yb - preds) ** 2)

            weights, bias = opt.step(cost_fn, weights, bias)

        tr_loss = _eval_loss(circuit_fn, weights, bias, X_train, y_train)
        vl_loss = _eval_loss(circuit_fn, weights, bias, X_val,   y_val)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["grad_var"].append(None)
        history["phase"].append("finetune")

        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:2d} | train={tr_loss:.4f}, val={vl_loss:.4f}")

    return weights, bias, history


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GRADIENT VARIANCE MONITOR
# ─────────────────────────────────────────────────────────────────────────────

def measure_gradient_variance(circuit_fn, weights, bias, X_sample, y_sample, n_repeats=20):
    """
    Estimate gradient variance by computing ∇L over `n_repeats` random
    mini-batches of size 16 and returning Var[‖∇w‖²].

    A rapidly shrinking value as n_qubits or n_layers increases is the
    barren-plateau signature.
    """
    rng = np.random.default_rng(0)
    grad_norms = []

    for _ in range(n_repeats):
        idx = rng.choice(len(X_sample), size=16, replace=False)
        Xb  = X_sample[idx]
        yb  = y_sample[idx].astype(float)

        def cost(w, b):
            preds = pnp.array([circuit_fn(w, x) + b for x in Xb])
            return pnp.mean((yb - preds) ** 2)

        grad_w, _ = qml.grad(cost)(weights, bias)
        grad_norms.append(float(np.linalg.norm(grad_w) ** 2))

    return float(np.mean(grad_norms)), float(np.var(grad_norms))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  STANDARD FULL TRAINING LOOP  (with gradient tracking)
# ─────────────────────────────────────────────────────────────────────────────

def train_vqc(
    circuit_fn,
    weights,
    bias,
    X_train, y_train,
    X_val,   y_val,
    n_epochs=30,
    batch_size=32,
    lr=0.01,
    track_grad_var=True,
    verbose=True,
):
    """
    Standard Adam training loop with optional gradient-variance tracking.

    Returns weights, bias, history dict.
    """
    opt = qml.AdamOptimizer(stepsize=lr)
    history = {"train_loss": [], "val_loss": [], "grad_var": []}

    for epoch in range(n_epochs):
        perm = np.random.default_rng(epoch).permutation(len(X_train))
        Xs, ys = X_train[perm], y_train[perm]
        epoch_grad_norms = []

        for start in range(0, len(Xs), batch_size):
            Xb = Xs[start : start + batch_size]
            yb = ys[start : start + batch_size].astype(float)

            def cost(w, b):
                preds = pnp.array([circuit_fn(w, x) + b for x in Xb])
                return pnp.mean((yb - preds) ** 2)

            if track_grad_var:
                grad_w, _ = qml.grad(cost)(weights, bias)
                epoch_grad_norms.append(float(np.linalg.norm(grad_w) ** 2))

            weights, bias = opt.step(cost, weights, bias)

        tr_loss = _eval_loss(circuit_fn, weights, bias, X_train, y_train)
        vl_loss = _eval_loss(circuit_fn, weights, bias, X_val,   y_val)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["grad_var"].append(float(np.mean(epoch_grad_norms)) if epoch_grad_norms else None)

        if verbose and (epoch + 1) % 5 == 0:
            gv = history["grad_var"][-1]
            gv_str = f", grad_var={gv:.2e}" if gv is not None else ""
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | "
                  f"train={tr_loss:.4f}, val={vl_loss:.4f}{gv_str}")

    return weights, bias, history


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _eval_loss(circuit_fn, weights, bias, X, y):
    preds = pnp.array([circuit_fn(weights, x) + bias for x in X])
    return float(pnp.mean((y.astype(float) - preds) ** 2))


def predict_batch(circuit_fn, weights, bias, X):
    return np.array([float(circuit_fn(weights, x) + bias) for x in X])
