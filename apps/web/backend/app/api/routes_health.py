from fastapi import APIRouter

from app.core.config import get_settings
from app.db.mongo import get_database
from autoshotv2.runtime import resolve_device

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    database_status = "ok"
    try:
        await get_database().command("ping")
    except Exception:
        database_status = "unavailable"

    checkpoint_exists = settings.autoshot_model_path.is_file()
    preferred_backend = (
        "baseline"
        if settings.autoshot_use_baseline or not checkpoint_exists
        else "phase2"
    )
    return {
        "status": "ok" if database_status == "ok" else "degraded",
        "database": database_status,
        "model": {
            "checkpoint": str(settings.autoshot_model_path),
            "checkpoint_exists": checkpoint_exists,
            "requested_device": settings.autoshot_device,
            "effective_device": resolve_device(settings.autoshot_device),
            "preferred_backend": preferred_backend,
        },
    }
