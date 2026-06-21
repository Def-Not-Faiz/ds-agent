import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def build_features(df: pd.DataFrame, target: str, options: dict | None = None):
    """
    Returns (X, y, feature_names, info).
    options (set by the agent), all optional:
      {"scale": False, "polynomial": False, "drop_cols": []}
    """
    options = options or {}
    info = []
    df = df.copy()

    # --- Drop explicitly requested columns ---
    for c in options.get("drop_cols", []):
        if c in df.columns:
            df = df.drop(columns=[c])
            info.append(f"Dropped column '{c}'.")

    # --- Separate target ---
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    y = df[target]
    X = df.drop(columns=[target])

    # --- Drop datetime columns (can't encode meaningfully without extra work) ---
    dt_cols = X.select_dtypes(include=["datetime64"]).columns.tolist()
    if dt_cols:
        X = X.drop(columns=dt_cols)
        info.append(f"Dropped datetime columns: {dt_cols}.")

    # --- Drop high-cardinality text columns ---
    for col in list(X.select_dtypes(include=["object"]).columns):
        unique_count = X[col].nunique()
        print(f"[features] {col}: {unique_count} unique values")
        if unique_count > 50:
            X = X.drop(columns=[col])
            info.append(f"Dropped high-cardinality text column '{col}' ({unique_count} unique).")

    # --- One-hot encode remaining categoricals ---
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        info.append(f"One-hot encoded: {cat_cols}.")

    # --- Fill missing values ---
    X = X.fillna(0)

    # --- Convert all columns to float (guards against bool/int mix after get_dummies) ---
    X = X.astype(float)

    # --- Reset index to avoid index mismatch with y ---
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # --- Polynomial interaction features (guarded: skip if too many columns) ---
    if options.get("polynomial", False):
        if X.shape[1] > 50:
            info.append(
                f"Skipped polynomial features: too many columns ({X.shape[1]} > 50). "
                "Would cause memory explosion."
            )
        else:
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            X_arr = poly.fit_transform(X)
            X = pd.DataFrame(X_arr, columns=poly.get_feature_names_out(X.columns))
            info.append(f"Added degree-2 interaction features ({X.shape[1]} total).")

    # --- Standard scaling ---
    if options.get("scale", False):
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        info.append("Scaled features (StandardScaler).")

    print(f"[features] Final feature matrix: {X.shape}")
    return X, y, list(X.columns), info