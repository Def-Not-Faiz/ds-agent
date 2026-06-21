import pandas as pd
from cleaning import clean_dataframe

df = pd.DataFrame({
    "age": [25, 30, None, 45, 200],
    "city": ["A", "B", "A", "C", None],
    "salary": [50000, 60000, 55000, None, 90000],
})
clean, log = clean_dataframe(df)
print("LOG:")
for x in log:
    print(" -", x)
print("MISSING AFTER:\n", clean.isna().sum())
