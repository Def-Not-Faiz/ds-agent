import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def _safe_name(col: str) -> str:
    """Convert column name to a safe filename slug."""
    return re.sub(r"[^\w]", "_", col).strip("_")


def generate_charts(df, out_dir="outputs", target=None) -> list:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Distribution histograms for numeric columns
    for col in numeric_cols[:6]:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution: {col}")
        p = os.path.join(out_dir, f"hist_{_safe_name(col)}.png")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths.append(p)

    # Correlation heatmap
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        p = os.path.join(out_dir, "correlation_heatmap.png")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths.append(p)

    if target and target in df.columns:
        # Target distribution (works for both classification and regression)
        plt.figure(figsize=(6, 4))
        if df[target].nunique() <= 20:
            sns.countplot(x=df[target])
            plt.title(f"Target Distribution: {target}")
        else:
            sns.histplot(df[target].dropna(), kde=True)
            plt.title(f"Target Distribution: {target}")
        p = os.path.join(out_dir, f"target_{_safe_name(target)}.png")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths.append(p)

        # Scatter plots of features vs target
        feature_cols = [c for c in numeric_cols if c != target]
        for col in feature_cols[:4]:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=df[col], y=df[target], alpha=0.3)
            plt.title(f"{col} vs {target}")
            p = os.path.join(out_dir, f"scatter_{_safe_name(col)}_vs_{_safe_name(target)}.png")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            paths.append(p)

    return paths