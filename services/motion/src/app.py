"""Local sentence-to-motion API. One inference worker prevents GPU/RAM contention."""
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from .paths import BODY_MODEL, CACHE, ROOT
from .schema import TranslationRequest, SCHEMA_VERSION
from .soke_provider import SOKEProvider, SOKE_ROOT as MODEL_ROOT, SOKE_REVISION as MODEL_REVISION
from .body import BodyModel

app = FastAPI(title="SignSense motion API", version="0.1.0")
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="motion")
jobs = {}
lock = threading.Lock()
provider = SOKEProvider()
body = BodyModel()
logger = logging.getLogger("uvicorn.error")

@app.middleware("http")
async def local_access(request: Request, call_next):
    # Research assets and generated motion must stay on the local machine.
    host = request.url.hostname
    if host not in {"localhost", "127.0.0.1", "testserver"}:
        return JSONResponse({"detail": "This research service is local-only."}, status_code=403)
    origin = request.headers.get("origin")
    if origin and origin not in {"http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:8000"}:
        return JSONResponse({"detail": "Origin is not allowed."}, status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response

def update(job_id, status, message, result=None):
    with lock:
        jobs[job_id].update(status=status, message=message, result=result)

def run_job(job_id, request, cache_key):
    directory = CACHE / cache_key
    try:
        cached = directory / "result.json"
        if cached.exists() and (directory / "positions.bin").exists():
            result = json.loads(cached.read_text())
            # Recheck old caches: numerical success previously allowed the
            # reported backward-arm failure to be labelled ready to play.
            import numpy as np
            result["diagnostics"] = body.validate_motion(np.load(directory / "motion133.npy", allow_pickle=False))
            result["cached"] = True
        else:
            update(job_id, "loading", "Loading the ASL research model…")
            provider.load()
            update(job_id, "generating", "Generating continuous ASL motion…")
            motion, meta = provider.generate(request.text, request.seed)
            diagnostics = body.validate_motion(motion)
            update(job_id, "preparing", "Preparing the native body, hands and facial motion…")
            rendering = body.prepare(motion, directory)
            result = {"schema": SCHEMA_VERSION, "text": request.text, "language": "ase",
                      "fps": meta["fps"], "duration": len(motion) / meta["fps"], "metadata": meta,
                      "rendering": rendering, "cached": False, "review": "unreviewed",
                      "diagnostics": diagnostics,
                      "assets": {"positions": f"/api/assets/{cache_key}/positions.bin",
                                 "rawMotion": f"/api/assets/{cache_key}/motion133.npy"}}
            directory.mkdir(parents=True, exist_ok=True)
            temp = directory / "result.json.tmp"
            temp.write_text(json.dumps(result))
            temp.replace(cached)
        update(job_id, "complete", "Ready to play. ASL accuracy is not yet reviewed.", result)
    except Exception as error:
        logger.exception("Translation job failed")
        # Local demo errors can include a useful dependency or device diagnosis.
        update(job_id, "failed", str(error)[:500])

@app.get("/api/health")
def health():
    files = ['model.safetensors', 'revision.json', 'config.json', 'map_ids.json', 'sentencepiece.bpe.model', 'mean133.npy', 'std133.npy', 'word2code.json']
    ready = BODY_MODEL.exists() and all((MODEL_ROOT / name).exists() for name in files)
    downloaded = sum((MODEL_ROOT / name).stat().st_size for name in files if (MODEL_ROOT / name).exists())
    total = downloaded if ready else 1900000000
    return {"ready": ready, "modelLoaded": provider.generator is not None, "model": "SOKE",
            "revision": MODEL_REVISION, "bodyReady": BODY_MODEL.exists(), "shardsReady": int((MODEL_ROOT / 'model.safetensors').exists()),
            "downloadedBytes": downloaded, "modelBytes": total, "review": "unreviewed"}

@app.get("/api/rig")
def rig():
    try:
        return body.rig()
    except (RuntimeError, ValueError) as error:
        raise HTTPException(503, str(error)) from error

@app.post("/api/translations", status_code=202)
def translate(request: TranslationRequest):
    if not health()["ready"]:
        raise HTTPException(503, "Model assets are still downloading. Check the status and try again.")
    cache_key = hashlib.sha256(json.dumps({"schema": SCHEMA_VERSION, "revision": MODEL_REVISION,
        "body": "smplx-native-soke-betas-v1", "pipeline": "soke-phrases-v2", **request.model_dump()}, sort_keys=True).encode()).hexdigest()
    with lock:
        # Identical in-flight requests reuse the original job.
        for job in jobs.values():
            if job["cacheKey"] == cache_key and job["status"] != "failed":
                return {k: v for k, v in job.items() if k != "cacheKey"}
        active = sum(j["status"] not in {"complete", "failed"} for j in jobs.values())
        if active >= 3:
            raise HTTPException(429, "The inference queue is full. Please wait for the current sentence.")
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "status": "queued", "message": "Waiting for the motion model…", "result": None, "cacheKey": cache_key}
        # Retain a bounded number of completed in-memory job records.
        for key in list(jobs):
            if len(jobs) < 100:
                break
            if jobs[key]["status"] in {"complete", "failed"}:
                del jobs[key]
        jobs[job_id] = job
    executor.submit(run_job, job_id, request, cache_key)
    return {k: v for k, v in job.items() if k != "cacheKey"}

@app.get("/api/translations/{job_id}")
def status(job_id: str):
    with lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Translation job not found.")
        return {k: v for k, v in job.items() if k != "cacheKey"}

@app.get("/api/assets/{cache_key}/{filename}")
def asset(cache_key: str, filename: str):
    if len(cache_key) != 64 or any(c not in "0123456789abcdef" for c in cache_key) or filename not in {"positions.bin", "motion133.npy"}:
        raise HTTPException(404)
    path = CACHE / cache_key / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="application/octet-stream")
