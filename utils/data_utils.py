"""
utils/data_utils.py
Shared data loading and preprocessing for VQC particle physics experiments.

HIGGS Dataset Feature Map (0-indexed CSV columns):
───────────────────────────────────────────────────
Col 0  : class label (1=signal, 0=background)
Col 1  : lepton pT           Col 2 : lepton eta
Col 3  : lepton phi          Col 4 : missing energy magnitude
Col 5  : missing energy phi   Col 6 : jet 1 pt
Col 7  : jet 1 eta           Col 8 : jet 1 phi
Col 9  : jet 1 b-tag         Col 10: jet 2 pt
Col 11 : jet 2 eta           Col 12: jet 2 phi
Col 13 : jet 2 b-tag         Col 14: jet 3 pt
Col 15 : jet 3 eta           Col 16: jet 3 phi
Col 17 : jet 3 b-tag         Col 18: jet 4 pt
Col 19 : jet 4 eta           Col 20: jet 4 phi
Col 21 : jet 4 b-tag
Col 22 : m_jj   (high-level) Col 23: m_jjj  (high-level)
Col 24 : m_lv   (high-level) Col 25: m_jlv  (high-level)
Col 26 : m_bb   (high-level) Col 27: m_wbb  (high-level)
Col 28 : m_wwbb (high-level)

NOTE: The paper (Blance & Spannowsky, JHEP 2021) uses a *different*
simulated dataset (Z' → tt̄ vs tt̄ background) with features pT,b1 and
E_T^miss. Since we use the UCI HIGGS dataset (H → ττ vs background),
we select features by correlation analysis to maximise discriminating
power for *this* dataset. Baseline analysis found columns 26 (m_bb)
and 4 (missing energy magnitude) are the top two uncorrelated features.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# ── Feature ranking by |correlation with label|, filtered for low
#    inter-feature correlation. Used as defaults across all experiments.
#    Derived from baseline notebook correlation analysis.
HIGGS_FEATURE_RANKING = [26, 4, 28, 1, 6, 13, 27, 25]
#                        m_bb, MET, m_wwbb, lep_pT, j1_pt, j2_btag, m_wbb, m_jlv

# Human-readable names for documentation / plots
HIGGS_FEATURE_NAMES = {
    1: "lepton pT",
    2: "lepton eta",
    3: "lepton phi",
    4: "missing energy mag.",
    5: "missing energy phi",
    6: "jet 1 pt",
    7: "jet 1 eta",
    8: "jet 1 phi",
    9: "jet 1 b-tag",
    10: "jet 2 pt",
    11: "jet 2 eta",
    12: "jet 2 phi",
    13: "jet 2 b-tag",
    14: "jet 3 pt",
    15: "jet 3 eta",
    16: "jet 3 phi",
    17: "jet 3 b-tag",
    18: "jet 4 pt",
    19: "jet 4 eta",
    20: "jet 4 phi",
    21: "jet 4 b-tag",
    22: "m_jj",
    23: "m_jjj",
    24: "m_lv",
    25: "m_jlv",
    26: "m_bb",
    27: "m_wbb",
    28: "m_wwbb",
}


def load_higgs(
    path="../data/HIGGS.csv.gz",
    n_samples=5000,
    n_features=2,
    feature_indices=None,
    scale_range=(0, np.pi),
    use_pca=False,
    random_state=42,
):
    """
    Load and preprocess the HIGGS dataset.

    Parameters
    ----------
    path : str
        Path to HIGGS.csv.gz
    n_samples : int
        Number of rows to load (paper uses 5000)
    n_features : int
        Number of features to select (2, 4, 6, or 8)
    feature_indices : list[int] or None
        Explicit column indices (0-indexed into the raw CSV, where col 0
        is the label). If None, selects the top n_features from
        HIGGS_FEATURE_RANKING (correlation-ranked).
    scale_range : tuple
        MinMaxScaler output range. Use (0, pi) for angle encoding.
    use_pca : bool
        If True, apply PCA to reduce to n_features dimensions instead
        of manually selecting columns.
    random_state : int

    Returns
    -------
    X_train, X_val, X_test : np.ndarray  (scaled)
    y_train, y_val, y_test : np.ndarray  (labels: -1 or +1)
    """
    data = pd.read_csv(path, header=None, compression="gzip", nrows=n_samples)

    # Column 0 is the label; columns 1–28 are features
    y_raw = data.iloc[:, 0].values
    y = np.where(y_raw == 1, 1, -1)  # map {0,1} → {-1,+1}

    # Feature selection
    if use_pca:
        X_raw = data.iloc[:, 1:].values
        scaler_raw = MinMaxScaler()
        X_raw = scaler_raw.fit_transform(X_raw)
        pca = PCA(n_components=n_features, random_state=random_state)
        X = pca.fit_transform(X_raw)
        print(
            f"PCA explained variance ({n_features} components): "
            f"{pca.explained_variance_ratio_.sum():.3f}"
        )
    else:
        if feature_indices is None:
            # Default: top-n from correlation-ranked features
            feature_indices = HIGGS_FEATURE_RANKING[:n_features]
        X = data.iloc[:, feature_indices].values
        feat_names = [HIGGS_FEATURE_NAMES.get(c, f"col{c}") for c in feature_indices]
        print(f"Selected features (cols {feature_indices}): {feat_names}")

    # Scale to encoding range
    scaler = MinMaxScaler(feature_range=scale_range)
    X = scaler.fit_transform(X)

    # Train / val / test  60 / 20 / 20
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=random_state
    )

    print(
        f"Dataset: {n_samples} samples | {X.shape[1]} features | "
        f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def binary_accuracy(y_true, y_pred_raw):
    """Threshold at 0 → {-1,+1}."""
    y_pred = np.where(y_pred_raw >= 0, 1, -1)
    return np.mean(y_pred == y_true)