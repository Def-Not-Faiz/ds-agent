import pandas as pd
from features import build_features
from training import detect_task, train_and_compare, feature_importance

df = pd.DataFrame({
    "age": [25, 30, 40, 45, 50, 28, 33, 60, 22, 48] * 5,
    "income": [30, 40, 55, 80, 90, 35, 50, 95, 28, 70] * 5,
    "bought": [0, 0, 1, 1, 1, 0, 1, 1, 0, 1] * 5,
})
X, y, names, _ = build_features(df, "bought")
task = detect_task(y)
res = train_and_compare(X, y, task)
print("TASK:", task, "| BEST:", res["best_model_name"], res["best_score"])
for r in res["results"]:
    print(r)
print("TOP FEATURES:", feature_importance(res["best_model"], names))
