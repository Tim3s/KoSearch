"""In-memory document indexing and method-independent search dispatch."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from app.models.embedder import Embedder
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex, model_index_path
from app.retrieval.hybrid import HybridRetriever
from app.schemas import Document, SearchResult


class SearchNotReady(RuntimeError):
    """Indicate that indexing or model initialization has not completed."""


class SearchService:
    """Own document state and route searches to BM25, dense, or hybrid retrievers."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        index_root: str | Path | None = None,
    ) -> None:
        """Create the service without loading an embedding model or indexing documents."""
        self.embedder = embedder
        self.index_root = Path(index_root) if index_root is not None else None
        self.documents: list[Document] = []
        self.bm25: BM25Index | None = None
        self.dense: DenseIndex | None = None
        self.bm25_indexing_seconds = 0.0
        self.dense_indexing_seconds: float | None = None

    def load_saved_index(self) -> bool:
        """Restore the model-specific dense index and rebuild its matching BM25 index."""
        if self.embedder is None or self.index_root is None:
            return False
        saved_path = model_index_path(self.index_root, self.embedder.model_name)
        if not (saved_path / "metadata.json").is_file():
            return False
        dense = DenseIndex.load(saved_path, self.embedder)
        lexical = BM25Index()
        lexical.index(dense.documents)
        self.documents = list(dense.documents)
        self.bm25 = lexical
        self.dense = dense
        return True

    @property
    def is_indexed(self) -> bool:
        """Return whether at least one document has been indexed for lexical search."""
        return self.bm25 is not None and self.bm25.document_count > 0

    def index_documents(self, documents: Sequence[Document]) -> dict[str, float | int | None]:
        """Replace the in-memory document collection and rebuild configured retrieval indexes."""
        if not documents:
            raise ValueError("at least one document is required")
        ids = [document.document_id for document in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("document IDs must be unique")
        started = time.perf_counter()
        lexical = BM25Index()
        lexical.index(documents)
        self.bm25_indexing_seconds = time.perf_counter() - started
        dense: DenseIndex | None = None
        dense_seconds: float | None = None
        if self.embedder is not None:
            dense = DenseIndex(self.embedder)
            dense_seconds = dense.build(documents)
            if self.index_root is not None:
                dense.save(model_index_path(self.index_root, self.embedder.model_name))
        self.documents = list(documents)
        self.bm25 = lexical
        self.dense = dense
        self.dense_indexing_seconds = dense_seconds
        return {
            "document_count": len(self.documents),
            "bm25_indexing_seconds": self.bm25_indexing_seconds,
            "dense_indexing_seconds": self.dense_indexing_seconds,
        }

    def search(
        self,
        query: str,
        *,
        method: str = "bm25",
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> list[SearchResult]:
        """Dispatch a query to a selected retrieval method using the shared result schema."""
        if not self.is_indexed or self.bm25 is None:
            raise SearchNotReady("documents have not been indexed")
        if method == "bm25":
            return self.bm25.search(query, top_k)
        if self.dense is None:
            raise SearchNotReady("dense model is disabled or has not indexed documents")
        if method == "dense":
            return self.dense.search(query, top_k)
        if method == "hybrid":
            return HybridRetriever(self.bm25, self.dense).search(query, top_k, alpha=alpha)
        raise ValueError(f"unsupported search method: {method}")
