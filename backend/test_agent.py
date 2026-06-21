import pandas as pd
from cleaning import clean_dataframe
from agent import run_agent

df = pd.DataFrame({
    "age": [25, 30, 40, 45, 50, 28, 33, 60, 22, 48] * 8,
    "income": [30, 40, 55, 80, 90, 35, 50, 95, 28, 70] * 8,
    "city": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"] * 8,
    "bought": [0, 0, 1, 1, 1, 0, 1, 1, 0, 1] * 8,
})
clean, _ = clean_dataframe(df)
out = run_agent(clean, "bought", threshold=0.95, max_rounds=4)
print("BEST:", out["best_model_name"], out["best_score"])
print("HISTORY:")
for h in out["history"]:
    print(" ", h)
