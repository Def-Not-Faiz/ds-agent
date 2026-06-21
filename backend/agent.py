import time
from features import build_features
from training import detect_task, train_and_compare, feature_importance, IMPROVEMENT_MENU
from llm_layer import choose_improvement

# How many consecutive non-improving rounds to allow before giving up early.
# Each round costs one LLM call (choose_improvement) + one retrain, so this
# caps wasted work once the agent stops finding real gains.
MAX_NO_IMPROVEMENT_STREAK = 1


def run_agent(clean_df, target: str, threshold: float = 0.8, max_rounds: int = 2) -> dict:
    """Reflect-and-iterate loop over a fixed menu of improvements."""
    feat_opts, model_opts = {}, {}
    tried, history = [], []
    no_improvement_streak = 0

    print(f"[agent] Starting run_agent | rows={len(clean_df)} | target='{target}' | threshold={threshold}")

    t0 = time.time()
    _, y_probe, _, _ = build_features(clean_df, target, {})
    task = detect_task(y_probe)
    print(f"[agent] Task detected: '{task}' ({time.time() - t0:.2f}s)")

    def train_with(fo, mo):
        print(f"[agent]   -> build_features | feat_opts={fo} | model_opts={mo}")
        t = time.time()
        X, y, names, finfo = build_features(clean_df, target, fo)
        print(f"[agent]   -> build_features done | X.shape={X.shape} ({time.time() - t:.2f}s)")

        print(f"[agent]   -> train_and_compare starting...")
        t = time.time()
        res = train_and_compare(X, y, task, mo)
        print(f"[agent]   -> train_and_compare done | best={res['best_model_name']} score={res['best_score']:.4f} ({time.time() - t:.2f}s)")

        res["feature_names"] = names
        res["feature_info"] = finfo
        return res

    print("[agent] Round 0: baseline training")
    best = train_with(feat_opts, model_opts)
    history.append({
        "round": 0,
        "action": "baseline",
        "reasoning": "initial training",
        "best_model": best["best_model_name"],
        "score": best["best_score"],
    })
    print(f"[agent] Baseline done | score={best['best_score']:.4f}")

    rnd = 1
    while best["best_score"] < threshold and rnd <= max_rounds:
        print(f"\n[agent] Round {rnd} | current best score={best['best_score']:.4f} | tried={tried}")

        print(f"[agent]   -> calling choose_improvement (LLM call)...")
        t = time.time()
        decision = choose_improvement({
            "task": task,
            "best_score": best["best_score"],
            "threshold": threshold,
            "round": rnd,
            "tried": tried,
            "menu": IMPROVEMENT_MENU,
            "last_results": best["results"],
        })
        print(f"[agent]   -> choose_improvement returned in {time.time() - t:.2f}s | decision={decision}")

        action = decision["action"]
        if action == "stop" or action in tried:
            print(f"[agent]   -> Stopping: action='{action}'")
            history.append({
                "round": rnd,
                "action": "stop",
                "reasoning": decision.get("reasoning", "no further gains"),
                "score": best["best_score"],
            })
            break
        tried.append(action)

        nfo, nmo = dict(feat_opts), dict(model_opts)
        if action == "scale_features":
            nfo["scale"] = True
        elif action == "add_interactions":
            nfo["polynomial"] = True
        elif action == "stronger_model":
            nmo["stronger_model"] = True
        elif action == "tune_random_forest":
            nmo["rf_estimators"] = 300
            nmo["rf_depth"] = 8

        print(f"[agent]   -> Training with action='{action}'")
        candidate = train_with(nfo, nmo)
        improved = candidate["best_score"] > best["best_score"]
        print(f"[agent]   -> Candidate score={candidate['best_score']:.4f} | improved={improved}")

        history.append({
            "round": rnd,
            "action": action,
            "reasoning": decision.get("reasoning", ""),
            "best_model": candidate["best_model_name"],
            "score": candidate["best_score"],
            "kept": improved,
        })

        if improved:
            best, feat_opts, model_opts = candidate, nfo, nmo
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1
            if no_improvement_streak >= MAX_NO_IMPROVEMENT_STREAK:
                print(
                    f"[agent]   -> Early stop: {no_improvement_streak} consecutive "
                    "non-improving round(s), not worth another LLM call + retrain."
                )
                history.append({
                    "round": rnd + 1,
                    "action": "stop",
                    "reasoning": "stopped early after consecutive non-improving rounds",
                    "score": best["best_score"],
                })
                rnd += 1
                break

        rnd += 1

    print(f"\n[agent] Loop finished | final best={best['best_model_name']} | score={best['best_score']:.4f}")

    print("[agent] Computing feature importance...")
    t = time.time()
    best["importance"] = feature_importance(best["best_model"], best["feature_names"])
    print(f"[agent] Feature importance done ({time.time() - t:.2f}s)")

    best["history"] = history
    best["task"] = task
    best_public = {k: v for k, v in best.items() if k != "best_model"}
    return best_public