"""Small request dependencies shared by API routers."""

from fastapi import HTTPException, Request

from app.services.search_service import SearchService


def get_search_service(request: Request) -> SearchService:
    """Return the single process-wide search service created during app startup."""
    service = getattr(request.app.state, "search_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="search service is not initialized")
    return service
