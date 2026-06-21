import pandas as pd
from cleaning import clean_dataframe
from visualize import generate_charts

df = pd.DataFrame({
    "age": [25, 30, 40, 45, 50, 28, 33],
    "salary": [50, 60, 55, 80, 90, 52, 58],
    "score": [1, 2, 2, 3, 4, 2, 3],
})
clean, _ = clean_dataframe(df)
paths = generate_charts(clean, out_dir="outputs", target="salary")
print("Saved:", paths)
