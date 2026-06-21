from agent import run_agent
import logging
import os
import re
import shutil
import threading
import time
import traceback
import uuid
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from profiling import profile_dataframe
from cleaning import clean_dataframe
from visualize import generate_charts
from llm_layer import identify_ml_task, generate_narrative, answer_question
from report import build_html_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data_science_agent")

app = FastAPI(title="Data Science Agent")

# Lock this down to your real frontend origin(s). "*" lets any website on the
# internet call this API from a user's browser - fine for a local demo,
# not fine once this is reachable outside localhost.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

LAST = {}

# job_id -> {"status", "step", "result", "error", "created_at"}
JOBS = {}

JOB_TTL_SECONDS = 60 * 60  # forget finished/failed jobs after an hour so JOBS doesn't grow forever

# Canonical step ids — MUST match the `id` field of PIPELINE_STEPS in the frontend's page.tsx.
# If you rename a step here, rename it there too, or progress tracking silently breaks again.
STEP_UPLOAD = "upload"
STEP_PROFILE = "profile"
STEP_CLEAN = "clean"
STEP_FEATURE = "feature"
STEP_TRAIN = "train"
STEP_EVALUATE = "evaluate"
STEP_EXPLAIN = "explain"
STEP_REPORT = "report"


def sanitize_filename(name: str) -> str:
    """Strip directory components and anything outside a safe charset, so a
    crafted filename like '../../etc/passwd' can't escape the uploads/ dir."""
    name = os.path.basename(name)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "upload.csv"


def sweep_old_jobs():
    now = time.time()
    expired = [
        jid for jid, job in JOBS.items()
        if job["status"] in ("done", "error") and now - job["created_at"] > JOB_TTL_SECONDS
    ]
    for jid in expired:
        del JOBS[jid]


@app.get("/")
def health():
    return {"status": "ok"}


def run_pipeline(job_id: str, path: str, prompt: str):
    """
    job["step"] is updated BEFORE each stage starts, using the canonical
    STEP_* ids above (not free-text sentences) so the frontend can match
    them against PIPELINE_STEPS and render per-step done/running/pending
    status correctly.

    If an exception is raised, job["step"] is deliberately left untouched
    in the except block - it still holds the last stage that was set,
    which tells you exactly where the pipeline got stuck.

    Timing is logged for every stage so a slowdown shows up immediately
    in the server logs instead of just "it's slow somewhere".
    """
    job = JOBS[job_id]
    t_pipeline_start = time.time()

    def log_stage_start(step_id: str):
        job["step"] = step_id
        job["_stage_start"] = time.time()
        logger.info("Job %s: START %s", job_id, step_id)

    def log_stage_end(step_id: str):
        elapsed = time.time() - job.get("_stage_start", time.time())
        logger.info("Job %s: END %s (%.2fs)", job_id, step_id, elapsed)

    try:
        log_stage_start(STEP_UPLOAD)
        df = pd.read_csv(path)
        logger.info("Job %s: CSV loaded %s", job_id, df.shape)
        log_stage_end(STEP_UPLOAD)

        log_stage_start(STEP_PROFILE)
        profile = profile_dataframe(df)
        ml_plan = identify_ml_task(profile, prompt)
        target = ml_plan.get("target_column")
        log_stage_end(STEP_PROFILE)

        log_stage_start(STEP_CLEAN)
        clean_df, log = clean_dataframe(df)
        log_stage_end(STEP_CLEAN)

        use_target = target if target and target in clean_df.columns else None

        if use_target:
            log_stage_start(STEP_TRAIN)
            agent_out = run_agent(clean_df, use_target)
            log_stage_end(STEP_TRAIN)

            job["step"] = STEP_EVALUATE
            # If run_agent already does train+eval internally, this just
            # marks the evaluate step as reached; nothing extra to compute.
            logger.info("Job %s: best=%s score=%s", job_id,
                        agent_out.get("best_model_name"), agent_out.get("best_score"))
        else:
            job["step"] = STEP_TRAIN
            agent_out = {
                "results": [],
                "history": [],
                "best_model_name": None,
                "best_score": None,
                "task": "unknown",
                "importance": [],
            }
            job["step"] = STEP_EVALUATE

        log_stage_start(STEP_FEATURE)
        charts = generate_charts(
            clean_df,
            out_dir="outputs",
            target=use_target,
        )
        log_stage_end(STEP_FEATURE)

        # Narrative generation (the LLM call) is the slowest single stage by
        # far on CPU-bound local models — often 60s+. Rather than block the
        # whole job on it, mark the job done with a placeholder narrative,
        # then fill in the real text in the background. The frontend polls
        # GET /narrative/{job_id} separately once the main result is in.
        log_stage_start(STEP_EXPLAIN)
        placeholder_narrative = "Generating insights..."
        job["narrative_status"] = "pending"
        log_stage_end(STEP_EXPLAIN)

        log_stage_start(STEP_REPORT)
        build_html_report(
            profile,
            log,
            ml_plan,
            agent_out,
            placeholder_narrative,
            charts,
            out="outputs/report.html",
        )
        log_stage_end(STEP_REPORT)

        result = {
            "profile": profile,
            "ml_plan": ml_plan,
            "cleaning_log": log,
            "agent": agent_out,
            "narrative": placeholder_narrative,
            "charts": [f"/outputs/{os.path.basename(c)}" for c in charts],
            "report_url": "/outputs/report.html",
            "preview": clean_df.head(10).to_dict(orient="records"),
            "preview_columns": list(clean_df.columns),
        }

        LAST.clear()
        LAST.update(result)

        job["result"] = result
        job["status"] = "done"
        job["step"] = STEP_REPORT
        logger.info("Job %s: TOTAL pipeline time %.2fs (narrative still pending)", job_id, time.time() - t_pipeline_start)

        # Kick off narrative generation in its own thread so the job above
        # is already "done" and visible to the user while this runs.
        narrative_thread = threading.Thread(
            target=run_narrative_job,
            args=(job_id, profile, log, ml_plan, agent_out, charts),
            daemon=True,
        )
        narrative_thread.start()

    except Exception as e:
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        logger.exception("Job %s failed at step: %s", job_id, job["step"])
    finally:
        # Don't keep raw uploaded data sitting on disk after it's been processed.
        try:
            os.remove(path)
        except OSError:
            pass


def run_narrative_job(job_id: str, profile, log, ml_plan, agent_out, charts):
    """Runs after the main pipeline already marked the job 'done'. Generates
    the real narrative, updates the job's result in place, then rebuilds the
    HTML report so it has the real text too."""
    job = JOBS.get(job_id)
    if not job:
        return
    t0 = time.time()
    try:
        narrative = generate_narrative(
            profile,
            log,
            ml_plan,
            {
                "task": agent_out["task"],
                "best_model": agent_out["best_model_name"],
                "best_score": agent_out["best_score"],
                "history": agent_out["history"],
                "importance": agent_out.get("importance", []),
            },
        )
        logger.info("Job %s: background narrative done (%.2fs)", job_id, time.time() - t0)

        if job.get("result"):
            job["result"]["narrative"] = narrative
        if LAST.get("ml_plan") == ml_plan:
            LAST["narrative"] = narrative
        job["narrative_status"] = "ready"

        build_html_report(
            profile,
            log,
            ml_plan,
            agent_out,
            narrative,
            charts,
            out="outputs/report.html",
        )
    except Exception as e:
        logger.exception("Job %s: background narrative failed", job_id)
        job["narrative_status"] = "error"
        if job.get("result"):
            job["result"]["narrative"] = "Could not generate narrative summary for this run."


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), prompt: str = Form("")):
    sweep_old_jobs()

    job_id = str(uuid.uuid4())
    safe_name = sanitize_filename(file.filename or "upload.csv")
    path = os.path.join("uploads", f"{job_id}_{safe_name}")

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {
        "status": "running",
        "step": STEP_UPLOAD,
        "result": None,
        "error": None,
        "created_at": time.time(),
        "narrative_status": "pending",
    }

    thread = threading.Thread(target=run_pipeline, args=(job_id, path, prompt), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"status": "error", "step": "unknown", "error": "Job not found"}
    return {"status": job["status"], "step": job["step"], "error": job["error"]}


@app.get("/result/{job_id}")
def result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"error": "Job not found"}
    if job["status"] != "done":
        return {"error": "Job not finished yet"}
    return job["result"]


@app.get("/narrative/{job_id}")
def narrative_status(job_id: str):
    """Lightweight poll endpoint for the background narrative generation.
    Frontend hits this after /result comes back, since /result returns
    immediately with a placeholder narrative while the real one finishes
    generating in the background."""
    job = JOBS.get(job_id)
    if not job:
        return {"status": "error", "narrative": None}
    return {
        "status": job.get("narrative_status", "pending"),
        "narrative": (job.get("result") or {}).get("narrative"),
    }


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(q: Question):
    if not LAST:
        return {"answer": "No analysis yet. Upload a CSV first."}
    context = {
        "ml_plan": LAST["ml_plan"],
        "best_model": LAST["agent"]["best_model_name"],
        "best_score": LAST["agent"]["best_score"],
        "model_comparison": LAST["agent"]["results"],
        "agent_history": LAST["agent"]["history"],
        "top_features": LAST["agent"].get("importance", []),
        "cleaning_log": LAST["cleaning_log"],
    }
    return {"answer": answer_question(context, q.question)}