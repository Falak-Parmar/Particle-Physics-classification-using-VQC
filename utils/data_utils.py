"""
utils/data_utils.py
Refactored data pipeline to ensure statistical rigor and prevent data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

HIGGS_FEATURE_RANKING = [26, 4, 28, 1, 6, 13, 27, 25]

HIGGS_FEATURE_NAMES = {
    1: "lepton pT", 2: "lepton eta", 3: "lepton phi", 4: "missing energy mag.",
    5: "missing energy phi", 6: "jet 1 pt", 7: "jet 1 eta", 8: "jet 1 phi",
    9: "jet 1 b-tag", 10: "jet 2 pt", 11: "jet 2 eta", 12: "jet 2 phi",
    13: "jet 2 b-tag", 14: "jet 3 pt", 15: "jet 3 eta", 16: "jet 3 phi",
    17: "jet 3 b-tag", 18: "jet 4 pt", 19: "jet 4 eta", 20: "jet 4 phi",
    21: "jet 4 b-tag", 22: "m_jj", 23: "m_jjj", 24: "m_lv", 25: "m_jlv",
    26: "m_bb", 27: "m_wbb", 28: "m_wwbb",
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
    Load and preprocess the HIGGS dataset with statistical rigor.
    
    Fixes:
    - Random sampling from a large buffer (prevents sequential bias).
    - Splitting before scaling (prevents data leakage).
    - Scaler fitted only on training data.
    """
    # Load a buffer to ensure we sample from different parts of the dataset
    # 100k is safe for memory and provides a good pool
    buffer_size = max(100000, n_samples * 2)
    data = pd.read_csv(path, header=None, compression="gzip", nrows=buffer_size)
    
    # Randomly sample n_samples
    data = data.sample(n=n_samples, random_state=random_state)

    # Column 0 is label (0 or 1); columns 1-28 are features
    y = data.iloc[:, 0].values.astype(int)
    
    if feature_indices is None:
        if n_features <= len(HIGGS_FEATURE_RANKING):
            feature_indices = HIGGS_FEATURE_RANKING[:n_features]
        else:
            feature_indices = list(range(1, n_features + 1))
            
    X = data.iloc[:, feature_indices].values

    # 1. SPLIT FIRST
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state
    )

    # 2. FIT SCALER ONLY ON TRAIN
    scaler = MinMaxScaler(feature_range=scale_range)
    X_train = scaler.fit_transform(X_train)
    
    # 3. TRANSFORM VAL/TEST USING TRAIN PARAMETERS
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Optional PCA (also fitted only on train)
    if use_pca:
        pca = PCA(n_components=n_features, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)

    print(f"Dataset: {n_samples} samples | {X_train.shape[1]} features")
    print(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def binary_accuracy(y_true, y_pred_raw):
    """
    Standard accuracy calculation.
    Expects y_true in {0, 1} or {-1, 1}.
    Thresholds raw predictions at 0.
    """
    y_true = np.array(y_true)
    y_pred_raw = np.array(y_pred_raw)
    
    # Handle both {0, 1} and {-1, 1}
    y_true_binary = np.where(y_true > 0, 1, 0)
    y_pred_binary = np.where(y_pred_raw >= 0, 1, 0)
    
    return np.mean(y_true_binary == y_pred_binary)
