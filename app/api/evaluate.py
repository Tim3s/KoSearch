"""Registered-dataset evaluation endpoint."""

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.dependencies import get_search_service
from app.api.schemas import EvaluateRequest
from app.data.autorag import DatasetLoadError, load_autorag_retrieval
from app.services.benchmark import BenchmarkRunner
from app.services.experiment_service import ExperimentStore
from app.services.search_service import SearchService

router = APIRouter(tags=["evaluation"])


@router.post("/evaluate")
def evaluate(
    request_body: EvaluateRequest,
    request: Request,
    service: SearchService = Depends(get_search_service),
) -> dict:
    """Evaluate only the registered AutoRAG dataset, then persist JSON and CSV results."""
    if any(method in {"dense", "hybrid"} for method in request_body.methods) and service.embedder is None:
        raise HTTPException(status_code=503, detail="dense model is disabled; set KOSEARCH_ENABLE_DENSE=true")
    try:
        dataset = load_autorag_retrieval()
    except DatasetLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        report = BenchmarkRunner(
            embedder=service.embedder,
            index_root=str(request.app.state.settings.data_dir / "indexes"),
        ).run(
            dataset,
            methods=request_body.methods,
            k_values=request_body.k_values,
            alpha_values=request_body.alpha_values,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    paths = request.app.state.experiment_store.save(report)
    return {**report, "files": paths}
