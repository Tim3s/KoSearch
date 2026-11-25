"""Document indexing endpoint."""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_search_service
from app.api.schemas import IndexDocumentsRequest
from app.schemas import Document
from app.services.search_service import SearchService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/index")
def index_documents(
    request: IndexDocumentsRequest,
    service: SearchService = Depends(get_search_service),
) -> dict[str, float | int | None | str]:
    """Replace the current in-memory document set and return index counts and timings."""
    documents = [Document(item.document_id, item.text, item.title) for item in request.documents]
    timings = service.index_documents(documents)
    return {"status": "indexed", **timings}
