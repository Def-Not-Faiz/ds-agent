import pandas as pd
import numpy as np


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Deterministic profiling. No LLM."""
    n_rows, n_cols = df.shape
    columns = []
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        elif n_unique <= 20:
            # Hard cap: anything with ≤20 unique values is categorical
            # Avoids the 0.05*n_rows trap that mislabels high-cardinality cols on large datasets
            kind = "categorical"
        else:
            kind = "text/high-cardinality"

        info = {
            "name": col,
            "dtype": str(s.dtype),
            "kind": kind,
            "missing": n_missing,
            "missing_pct": round(100 * n_missing / n_rows, 2),
            "unique": n_unique,
        }
        if kind == "numeric":
            info.update({
                "mean": _safe(s.mean()),
                "median": _safe(s.median()),
                "std": _safe(s.std()),
                "min": _safe(s.min()),
                "max": _safe(s.max()),
                "skew": _safe(s.skew()),
                "outliers_iqr": int(_count_outliers_iqr(s)),
            })
        columns.append(info)

    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr().round(3).to_dict() if numeric_df.shape[1] >= 2 else {}

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_duplicates": int(df.duplicated().sum()),
        "columns": columns,
        "correlations": correlations,
    }


def _safe(x):
    if pd.isna(x):
        return None
    return round(float(x), 4)


def _count_outliers_iqr(s: pd.Series) -> int:
    s = s.dropna()
    if len(s) < 4:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum())