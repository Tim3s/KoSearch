"""Search endpoint for BM25, dense, and hybrid methods."""

import time

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.dependencies import get_search_service
from app.api.schemas import SearchHit, SearchRequest, SearchResponse
from app.services.search_service import SearchNotReady, SearchService

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    http_request: Request,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Run the requested retrieval method and return ranked documents with elapsed time."""
    if request.top_k > http_request.app.state.settings.max_top_k:
        raise HTTPException(
            status_code=422,
            detail=f"top_k cannot exceed {http_request.app.state.settings.max_top_k}",
        )
    started = time.perf_counter()
    try:
        results = service.search(
            request.query,
            method=request.method,
            top_k=request.top_k,
            alpha=request.alpha,
        )
    except SearchNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    hits = [
        SearchHit(
            document_id=result.document_id,
            title=result.title,
            text=result.text,
            score=result.score,
            rank=rank,
        )
        for rank, result in enumerate(results, start=1)
    ]
    return SearchResponse(
        query=request.query,
        method=request.method,
        latency_ms=(time.perf_counter() - started) * 1000,
        results=hits,
    )
