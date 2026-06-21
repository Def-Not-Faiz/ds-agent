import os
from datetime import datetime


def build_html_report(profile, cleaning_log, ml_plan, agent_out, narrative, chart_paths,
                      out="outputs/report.html"):

    # --- Charts ---
    charts_html = "".join(
        f'<div class="chart"><img src="{os.path.basename(p)}" style="max-width:700px"></div>'
        for p in chart_paths
    )

    # --- Column table ---
    cols_rows = "".join(
        f"<tr><td>{c['name']}</td><td>{c['kind']}</td>"
        f"<td>{c['missing_pct']}%</td><td>{c['unique']}</td></tr>"
        for c in profile["columns"]
    )

    # --- Model comparison table ---
    if agent_out.get("results"):
        model_head = "".join(f"<th>{k}</th>" for k in agent_out["results"][0].keys())
        model_rows = "".join(
            "<tr>" + "".join(f"<td>{v}</td>" for v in r.values()) + "</tr>"
            for r in agent_out["results"]
        )
        model_table = f"<table><tr>{model_head}</tr>{model_rows}</table>"
    else:
        model_table = "<p>No model trained (no target identified).</p>"

    # --- Agent history table ---
    hist_rows = "".join(
        f"<tr><td>{h.get('round', '')}</td><td>{h.get('action', '')}</td>"
        f"<td>{h.get('score', '')}</td><td>{h.get('reasoning', '')}</td></tr>"
        for h in agent_out.get("history", [])
    )

    # --- Feature importance table ---
    imp_rows = "".join(
        f"<tr><td>{i['feature']}</td><td>{i['importance']}</td></tr>"
        for i in agent_out.get("importance", [])
    )

    # --- Cleaning log ---
    cleaning_items = "".join(f"<li>{x}</li>" for x in cleaning_log)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Data Science Agent Report</title>
  <style>
    body {{ font-family: system-ui; max-width: 920px; margin: 40px auto; line-height: 1.6; padding: 0 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
    td, th {{ border: 1px solid #ccc; padding: 6px; text-align: left; }}
    h2 {{ margin-top: 34px; border-bottom: 1px solid #eee; padding-bottom: 4px; }}
    pre {{ background: #f4f4f4; padding: 12px; border-radius: 6px; white-space: pre-wrap; }}
    .chart {{ margin: 18px 0; }}
    .badge {{ background: #e8f4fd; padding: 2px 8px; border-radius: 4px; font-size: 0.9em; }}
  </style>
</head>
<body>

<h1>Automated Data Science Agent — Report</h1>
<p><em>Generated {datetime.now():%Y-%m-%d %H:%M}</em></p>

<h2>Overview</h2>
<p>{narrative}</p>

<h2>Best Model</h2>
<p>
  <b>{agent_out.get('best_model_name', 'N/A')}</b>
  <span class="badge">task: {agent_out.get('task', 'unknown')}</span>
  <span class="badge">score: {agent_out.get('best_score', 'N/A')}</span>
</p>

<h2>Model Comparison</h2>
{model_table}

<h2>Agent Improvement History</h2>
<table>
  <tr><th>Round</th><th>Action</th><th>Score</th><th>Reasoning</th></tr>
  {hist_rows}
</table>

<h2>Top Features</h2>
<table>
  <tr><th>Feature</th><th>Importance</th></tr>
  {imp_rows}
</table>

<h2>Dataset Columns</h2>
<table>
  <tr><th>Name</th><th>Kind</th><th>Missing %</th><th>Unique Values</th></tr>
  {cols_rows}
</table>

<h2>Cleaning Log</h2>
<ul>{cleaning_items}</ul>

<h2>ML Plan</h2>
<pre>{ml_plan}</pre>

<h2>Visualizations</h2>
{charts_html}

</body>
</html>"""

    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out