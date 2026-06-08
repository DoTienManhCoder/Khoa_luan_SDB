import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.db.mongo import get_database
from app.services.job_serializer import serialize_job
from app.services.storage_service import (
    delete_cloudinary_assets,
    publish_asset,
    remove_job_files,
    safe_video_extension,
    upload_dir,
)
from app.worker.runner import process_job

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _clamp_opt(value: float | None, lo: float, hi: float) -> float | None:
    if value is None:
        return None
    return max(lo, min(float(value), hi))


@router.post("/from-upload")
async def create_job_from_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sensitivity: str = Form("medium"),
    min_scene_duration_sec: float = Form(0.5),
    backend: str = Form("auto"),
    temperature: float | None = Form(None),
    sigma: float | None = Form(None),
    threshold: float | None = Form(None),
) -> dict:
    settings = get_settings()
    sensitivity = sensitivity if sensitivity in {"low", "medium", "high"} else settings.default_sensitivity
    min_scene_duration_sec = max(0.1, min(float(min_scene_duration_sec), 10.0))
    backend = backend if backend in {"auto", "phase2", "baseline"} else "auto"
    temperature = _clamp_opt(temperature, 0.05, 5.0)
    sigma = _clamp_opt(sigma, 0.0, 10.0)
    threshold = _clamp_opt(threshold, 0.0, 1.0)

    try:
        extension = safe_video_extension(file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = f"job_{uuid4().hex[:16]}"
    target_dir = upload_dir(job_id)
    video_path = target_dir / f"input{extension}"

    size_bytes = 0
    async with aiofiles.open(video_path, "wb") as handle:
        while chunk := await file.read(1024 * 1024):
            size_bytes += len(chunk)
            if size_bytes > settings.max_upload_bytes:
                remove_job_files(job_id)
                raise HTTPException(status_code=413, detail=f"Upload limit is {settings.max_upload_mb}MB")
            await handle.write(chunk)

    if size_bytes == 0:
        remove_job_files(job_id)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    video_asset = await asyncio.to_thread(publish_asset, video_path, job_id, "input", "video")
    cloudinary_assets = [video_asset] if "public_id" in video_asset else []

    now = datetime.now(timezone.utc)
    document = {
        "_id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0.05,
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "expires_at": now + timedelta(hours=settings.job_ttl_hours),
        "input": {
            "original_name": Path(file.filename or "video").name,
            "size_bytes": size_bytes,
            "content_type": file.content_type or "application/octet-stream",
        },
        "processing": {
            "model": "pending",
            "sensitivity": sensitivity,
            "min_scene_duration_sec": min_scene_duration_sec,
            "backend": backend,
            "temperature": temperature,
            "sigma": sigma,
            "threshold": threshold,
        },
        "storage": {
            "video_url": video_asset["url"],
        },
        "cloudinary": {
            "enabled": settings.use_cloudinary,
            "folder": settings.cloudinary_folder,
            "assets": cloudinary_assets,
        },
        "summary": None,
        "boundaries": [],
        "scenes": [],
        "exports": {},
        "artifacts": {},
        "error": None,
        "internal": {
            "video_path": str(video_path),
        },
    }

    db = get_database()
    await db.jobs.insert_one(document)
    background_tasks.add_task(process_job, job_id)
    return serialize_job(document)


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict:
    db = get_database()
    job = await db.jobs.find_one({"_id": job_id})
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return serialize_job(job)


@router.get("/{job_id}/scenes")
async def get_job_scenes(job_id: str) -> dict:
    db = get_database()
    job = await db.jobs.find_one({"_id": job_id}, {"scenes": 1})
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "scenes": job.get("scenes", [])}


@router.get("/{job_id}/boundaries")
async def get_job_boundaries(job_id: str) -> dict:
    db = get_database()
    job = await db.jobs.find_one({"_id": job_id}, {"boundaries": 1})
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "boundaries": job.get("boundaries", [])}


@router.get("/{job_id}/exports/{kind}")
async def download_export(job_id: str, kind: str) -> FileResponse:
    names = {
        "json": ("result.json", "application/json"),
        "csv": ("scenes.csv", "text/csv"),
        "txt": ("summary.txt", "text/plain"),
    }
    if kind not in names:
        raise HTTPException(status_code=404, detail="Export not found")

    filename, media_type = names[kind]
    path = get_settings().storage_dir / "jobs" / job_id / "exports" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export not ready")
    return FileResponse(path, media_type=media_type, filename=filename)


@router.delete("/{job_id}")
async def delete_job(job_id: str) -> dict[str, str]:
    db = get_database()
    job = await db.jobs.find_one({"_id": job_id})
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    await asyncio.to_thread(delete_cloudinary_assets, job)
    result = await db.jobs.delete_one({"_id": job_id})
    remove_job_files(job_id)
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "deleted"}
