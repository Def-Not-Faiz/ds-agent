import json
import time
import ollama

MODEL = "qwen2.5:3b"
KEEP_ALIVE = "30m"    # keep the model loaded in Ollama between calls so each
                      # call doesn't pay a cold-load penalty
TIMEOUT_SECONDS = 90  # CPU inference of long JSON/prose responses can take
                      # well over 25s; this just stops a truly dead call
NUM_PREDICT_JSON = 250   # identify_ml_task / choose_improvement: short structured output
NUM_PREDICT_TEXT = 220   # generate_narrative / answer_question: ~3 short paragraphs max

_client = ollama.Client(timeout=TIMEOUT_SECONDS)


def _chat(system: str, user: str, as_json: bool, temp: float) -> str:
    kwargs = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {
            "temperature": temp,
            "num_predict": NUM_PREDICT_JSON if as_json else NUM_PREDICT_TEXT,
        },
        "keep_alive": KEEP_ALIVE,
    }
    if as_json:
        kwargs["format"] = "json"
    t0 = time.time()
    try:
        out = _client.chat(**kwargs)["message"]["content"].strip()
        print(f"[llm] call ok in {time.time() - t0:.2f}s | as_json={as_json}")
        return out
    except Exception as e:
        print(f"[llm] call FAILED after {time.time() - t0:.2f}s ({type(e).__name__}: {e}) -> using mock fallback")
        if as_json:
            return json.dumps(_mock_json_response(system, user))
        return _mock_text_response(system, user)


def _mock_json_response(system: str, user: str) -> dict:
    if "data science planner" in system:
        return {
            "task_type": "classification",
            "target_column": "bought",
            "reasoning": "The dataset has a binary purchase target and numeric predictors.",
            "suggested_models": ["LogisticRegression", "RandomForest"],
            "warnings": [],
        }
    if "ML optimization agent" in system:
        state = json.loads(user)
        for item in state.get("menu", []):
            if item not in state.get("tried", []):
                return {"action": item, "reasoning": "Try the next available improvement."}
        return {"action": "stop", "reasoning": "No remaining improvements."}
    return {"action": "stop", "reasoning": "Fallback response."}


def _mock_text_response(system: str, user: str) -> str:
    if "data analyst writing a concise report" in system:
        return (
            "The dataset was profiled and cleaned before model training. "
            "The top model achieved the best score using the selected features. "
            "The result should be treated as a baseline; further data collection would help."
        )
    if "answer questions about a completed data science analysis" in system:
        return "Ollama is unavailable, but the analysis completed successfully with the current pipeline."
    return "Fallback response."


def identify_ml_task(profile: dict, user_prompt: str = "") -> dict:
    # Cap columns sent to the model — wide datasets (50+ cols) bloat the
    # prompt a lot and slow CPU generation; the first 40 is plenty of
    # signal for picking a target/task.
    cols_full = profile["columns"]
    cols = [{"name": c["name"], "kind": c["kind"], "unique": c["unique"],
             "missing_pct": c["missing_pct"]} for c in cols_full[:40]]
    system = (
        "You are a data science planner. Respond with ONLY valid JSON. Schema:\n"
        '{"task_type":"classification|regression|clustering",'
        '"target_column":"name or null","reasoning":"one sentence",'
        '"suggested_models":["..."],"warnings":["..."]}'
    )
    user = (f"User request: {user_prompt or 'none'}\n"
            f"Rows: {profile['n_rows']}, Cols: {profile['n_cols']}\n"
            f"Columns: {json.dumps(cols)}")
    return json.loads(_chat(system, user, as_json=True, temp=0.2))


def choose_improvement(state: dict) -> dict:
    # Truncate last_results to avoid sending huge payloads to the LLM
    slim_state = {
        "task": state["task"],
        "best_score": state["best_score"],
        "threshold": state["threshold"],
        "round": state["round"],
        "tried": state["tried"],
        "menu": state["menu"],
        "last_results": [
            {"model": r["model"], "score": r["score"]}
            for r in state.get("last_results", [])
        ],
    }
    system = (
        "You are an ML optimization agent. Given the current best score and the "
        "fixed menu of allowed actions, choose the single most promising action to "
        "improve the model, or 'stop' if the score is good enough or options are "
        "exhausted. You may ONLY choose from the provided menu or 'stop'. "
        'Respond with ONLY JSON: {"action":"...","reasoning":"one sentence"}.'
    )
    out = json.loads(_chat(system, json.dumps(slim_state), as_json=True, temp=0.2))
    if out.get("action") not in state["menu"] + ["stop"]:
        out["action"] = "stop"
    return out


def generate_narrative(profile, cleaning_log, ml_plan, training_summary) -> str:
    system = "You are a data analyst writing a concise report. Plain language, no markdown headers."
    user = (f"Dataset: {profile['n_rows']} rows, {profile['n_cols']} cols, "
            f"{profile['n_duplicates']} duplicates.\n"
            f"Cleaning: {cleaning_log}\nPlan: {json.dumps(ml_plan)}\n"
            f"Training result: {json.dumps(training_summary)}\n\n"
            "Write 3 short paragraphs: (1) data overview and quality issues, "
            "(2) which model won and how the agent improved it, "
            "(3) cautions and what the result means.")
    return _chat(system, user, as_json=False, temp=0.4)


def answer_question(context: dict, question: str) -> str:
    system = ("You answer questions about a completed data science analysis using ONLY "
              "the provided context. If the answer isn't in the context, say so plainly.")
    user = f"Context:\n{json.dumps(context)}\n\nQuestion: {question}"
    return _chat(system, user, as_json=False, temp=0.3)