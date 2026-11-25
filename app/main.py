"""FastAPI application factory and health endpoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import documents, evaluate, experiments, search
from app.config import get_settings
from app.models.embedder import load_embedder
from app.services.experiment_service import ExperimentStore
from app.services.search_service import SearchService


def create_app() -> FastAPI:
    """Create the API app and load an embedding model once when dense mode is enabled."""
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize shared services once and attach them to the application state."""
        embedder = None
        if settings.enable_dense:
            embedder = load_embedder(
                settings.model_name,
                batch_size=settings.batch_size,
                device=settings.device,
            )
        app.state.settings = settings
        app.state.search_service = SearchService(
            embedder=embedder,
            index_root=settings.data_dir / "indexes",
        )
        if embedder is not None:
            app.state.search_service.load_saved_index()
        app.state.experiment_store = ExperimentStore(settings.experiments_dir)
        yield

    application = FastAPI(title="KoSearch Lab", version="0.1.0", lifespan=lifespan)
    application.include_router(documents.router)
    application.include_router(search.router)
    application.include_router(evaluate.router)
    application.include_router(experiments.router)

    @application.get("/health", tags=["health"])
    def health() -> dict[str, object]:
        """Report service, index, model, and device state without probing private paths."""
        service = getattr(application.state, "search_service", None)
        return {
            "status": "ok",
            "index_loaded": bool(service and service.is_indexed),
            "model": service.embedder.model_name if service and service.embedder else None,
            "device": service.embedder.device if service and service.embedder else None,
        }

    return application


app = create_app()
