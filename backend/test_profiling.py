import json
import pandas as pd
from profiling import profile_dataframe

df = pd.DataFrame({
    "age": [25, 30, None, 45, 200],
    "city": ["A", "B", "A", "C", "B"],
    "salary": [50000, 60000, 55000, None, 90000],
})
print(json.dumps(profile_dataframe(df), indent=2, default=str))
