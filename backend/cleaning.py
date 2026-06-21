import pandas as pd
import numpy as np


_JUNK_VALUES = {"ERROR", "UNKNOWN", "N/A", "NA", "NULL", "none", "None", ""}


def clean_dataframe(df: pd.DataFrame, target: str | None = None, strategy: dict | None = None) -> tuple[pd.DataFrame, list]:
    """Deterministic cleaning. Returns (clean_df, log)."""
    strategy = strategy or {}
    log = []
    df = df.copy()

    # --- Step 1: Replace junk sentinel values with NaN ---
    df = df.replace(list(_JUNK_VALUES), np.nan)
    log.append("Replaced sentinel values (ERROR, UNKNOWN, N/A, etc.) with NaN.")

    # --- Step 2: Coerce columns that look numeric to float ---
    # Use select_dtypes(include="object") AND also check pd.StringDtype
    # because newer pandas uses dtype=str (pd.StringDtype) not dtype=object
    for col in df.columns:
        col_dtype = df[col].dtype
        is_string_col = (
            col_dtype == object
            or isinstance(col_dtype, pd.StringDtype)
            or str(col_dtype) == "string"
            or str(col_dtype) == "str"
        )
        if not is_string_col:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].notna().sum()
        converted_non_null = converted.notna().sum()
        if non_null > 0 and (converted_non_null / non_null) >= 0.8:
            df[col] = converted
            log.append(f"Coerced '{col}' from string to numeric.")

    # --- Step 3: Drop duplicates ---
    if strategy.get("drop_duplicates", True):
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)
        if removed:
            log.append(f"Removed {removed} duplicate rows.")

    # --- Step 4: Fill missing values ---
    num_fill = strategy.get("missing_numeric", "median")
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            val = df[col].median() if num_fill == "median" else df[col].mean()
            df[col] = df[col].fillna(val)
            log.append(f"Filled missing in '{col}' with {num_fill} ({round(float(val), 3)}).")
        else:
            mode = df[col].mode()
            val = mode.iloc[0] if not mode.empty else "UNKNOWN"
            df[col] = df[col].fillna(val)
            log.append(f"Filled missing in '{col}' with mode ('{val}').")

    # --- Step 5: Cap outliers (skip target column) ---
    if strategy.get("outlier_method", "iqr") == "iqr":
        for col in df.select_dtypes(include=[np.number]).columns:
            if target and col == target:
                continue
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n:
                df[col] = df[col].clip(lo, hi)
                log.append(f"Capped {n} outliers in '{col}' to IQR bounds.")

    return df, log