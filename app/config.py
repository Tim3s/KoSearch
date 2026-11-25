"""Environment-backed application settings with conservative local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration shared by the API and command-line tools."""

    model_name: str = "intfloat/multilingual-e5-base"
    batch_size: int = 32
    device: str = "auto"
    enable_dense: bool = False
    data_dir: Path = Path("data")
    experiments_dir: Path = Path("experiments")
    max_top_k: int = 100


def get_settings() -> Settings:
    """Read supported KoSearch environment variables into validated settings."""
    batch_size = int(os.getenv("KOSEARCH_BATCH_SIZE", "32"))
    max_top_k = int(os.getenv("KOSEARCH_MAX_TOP_K", "100"))
    if batch_size < 1 or max_top_k < 1:
        raise ValueError("KOSEARCH_BATCH_SIZE and KOSEARCH_MAX_TOP_K must be positive")
    return Settings(
        model_name=os.getenv("KOSEARCH_MODEL", "intfloat/multilingual-e5-base"),
        batch_size=batch_size,
        device=os.getenv("KOSEARCH_DEVICE", "auto"),
        enable_dense=os.getenv("KOSEARCH_ENABLE_DENSE", "false").lower() in {"1", "true", "yes"},
        data_dir=Path(os.getenv("KOSEARCH_DATA_DIR", "data")),
        experiments_dir=Path(os.getenv("KOSEARCH_EXPERIMENTS_DIR", "experiments")),
        max_top_k=max_top_k,
    )
