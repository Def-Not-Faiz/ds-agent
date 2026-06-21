import pandas as pd
from profiling import profile_dataframe
from cleaning import clean_dataframe
from agent import run_agent
from report import build_html_report

# Create a simple dataset
df = pd.DataFrame({
    "age": [25, 30, 40, 45, 50, 28, 33, 60, 22, 48],
    "income": [30, 40, 55, 80, 90, 35, 50, 95, 28, 70],
    "bought": [0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
})
profile = profile_dataframe(df)
clean_df, log = clean_dataframe(df)
agent_out = run_agent(clean_df, "bought", threshold=0.7, max_rounds=1)
html_path = build_html_report(profile, log, {"task_type":"classification","target_column":"bought"}, agent_out, "This is a narrative.", ["outputs/correlation_heatmap.png"], out="outputs/test_report.html")
print(html_path)
