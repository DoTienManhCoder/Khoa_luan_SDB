from functools import lru_cache

from app.core.config import get_settings
from autoshotv2.runtime import AutoShotRuntime


@lru_cache(maxsize=1)
def get_runtime() -> AutoShotRuntime:
    settings = get_settings()
    return AutoShotRuntime(settings.autoshot_model_path, settings.autoshot_device)


def clear_runtime_cache() -> None:
    get_runtime.cache_clear()
