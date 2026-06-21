import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error

IMPROVEMENT_MENU = [
    "scale_features",
    "add_interactions",
    "stronger_model",
    "tune_random_forest",
]


def detect_task(y) -> str:
    """Infer classification vs regression from the target."""
    # String target → always classification
    if y.dtype == object:
        return "classification"

    nun = y.nunique()

    # Float target → almost always regression
    if y.dtype in [float, np.float32, np.float64]:
        # Only treat as classification if very few unique float values (e.g. 0.0 / 1.0)
        return "classification" if nun <= 10 else "regression"

    # Integer target: use a hard cap of 20 unique values for classification
    # Avoids the 0.05 * len(y) trap on large datasets
    return "classification" if nun <= 20 else "regression"


def _models(task: str, opts: dict):
    if task == "classification":
        m = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(
                n_estimators=opts.get("rf_estimators", 100),
                max_depth=opts.get("rf_depth", None), random_state=42),
        }
        if opts.get("stronger_model"):
            m["GradientBoosting"] = GradientBoostingClassifier(random_state=42)
    else:
        m = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "RandomForest": RandomForestRegressor(
                n_estimators=opts.get("rf_estimators", 100),
                max_depth=opts.get("rf_depth", None), random_state=42),
        }
        if opts.get("stronger_model"):
            m["GradientBoosting"] = GradientBoostingRegressor(random_state=42)
    return m


def train_and_compare(X, y, task: str, opts: dict | None = None) -> dict:
    """Train the roster, return ranked results + the best model object."""
    opts = opts or {}
    strat = y if task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=strat)

    results, fitted = [], {}
    for name, model in _models(task, opts).items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        if task == "classification":
            row = {
                "model": name,
                "accuracy": round(float(accuracy_score(y_te, pred)), 4),
                "f1": round(float(f1_score(y_te, pred, average="weighted")), 4),
            }
            if y.nunique() == 2:
                try:
                    proba = model.predict_proba(X_te)[:, 1]
                    row["roc_auc"] = round(float(roc_auc_score(y_te, proba)), 4)
                except Exception:
                    pass
            row["score"] = row["f1"]
        else:
            rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
            row = {
                "model": name,
                "r2": round(float(r2_score(y_te, pred)), 4),
                "rmse": round(rmse, 4),
                "mae": round(float(mean_absolute_error(y_te, pred)), 4),
            }
            row["score"] = row["r2"]
        results.append(row)
        fitted[name] = model

    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]
    return {
        "task": task,
        "results": results,
        "best_model_name": best["model"],
        "best_score": best["score"],
        "best_model": fitted[best["model"]],
    }


def feature_importance(best_model, feature_names) -> list:
    """Top features if the model exposes them."""
    if hasattr(best_model, "feature_importances_"):
        pairs = sorted(zip(feature_names, best_model.feature_importances_),
                       key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(i), 4)} for f, i in pairs[:10]]
    if hasattr(best_model, "coef_"):
        coefs = np.ravel(best_model.coef_)
        pairs = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
        return [{"feature": f, "importance": round(float(c), 4)} for f, c in pairs[:10]]
    return []