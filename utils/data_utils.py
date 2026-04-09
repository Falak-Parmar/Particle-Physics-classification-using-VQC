"""
utils/data_utils.py
Shared data loading and preprocessing for VQC particle physics experiments.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


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
        Explicit column indices to select (1-indexed into the data).
        If None, selects the first n_features physics columns.
        Paper uses columns 4 and 6 (0-indexed into raw data, i.e. cols 4,6).
    scale_range : tuple
        MinMaxScaler output range. Use (0, pi) for angle encoding.
    use_pca : bool
        If True, apply PCA to reduce to n_features dimensions instead
        of manually selecting columns. Useful when n_features > 2.
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
            # Paper default: columns 4 and 6 (0-indexed into raw CSV)
            # For larger n_features we pick the first n low-level features
            default_cols = [4, 6, 7, 8, 1, 2, 3, 5]  # sensible ordering
            feature_indices = default_cols[:n_features]
        X = data.iloc[:, feature_indices].values

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
        f"Dataset: {n_samples} samples | {n_features} features | "
        f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def binary_accuracy(y_true, y_pred_raw):
    """Threshold at 0 → {-1,+1}."""
    y_pred = np.where(y_pred_raw >= 0, 1, -1)
    return np.mean(y_pred == y_true)
