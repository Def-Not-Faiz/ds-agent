import pandas as pd
from features import build_features

df = pd.DataFrame({
    "age": [25, 30, 40, 45, 50],
    "city": ["A", "B", "A", "C", "B"],
    "bought": [1, 0, 1, 0, 1],
})
X, y, names, info = build_features(df, "bought", {"scale": True})
print("FEATURES:", names)
print("INFO:", info)
print(X.head())
