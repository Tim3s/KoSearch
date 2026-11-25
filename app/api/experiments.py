"""Experiment result retrieval endpoint."""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("/{experiment_id}")
def get_experiment(experiment_id: str, request: Request) -> dict:
    """Read a stored experiment using the path-safe experiment store."""
    try:
        return request.app.state.experiment_store.get(experiment_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="experiment not found") from exc
