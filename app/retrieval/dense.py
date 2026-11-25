"""Normalized dense vector search with FAISS and a NumPy fallback."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from app.models.embedder import Embedder
from app.schemas import Document, SearchResult


class DenseIndex:
    """Store normalized passage vectors and rank them by cosine similarity."""

    def __init__(self, embedder: Embedder) -> None:
        """Create an empty index bound to one embedding model identity."""
        self.embedder = embedder
        self.model_name = embedder.model_name
        self.documents: list[Document] = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.indexing_seconds = 0.0
        self.encoding_seconds = 0.0
        self._faiss_index = None

    @property
    def document_count(self) -> int:
        """Return the number of indexed document vectors."""
        return len(self.documents)

    @property
    def backend(self) -> str:
        """Report whether searches use FAISS or the exact NumPy fallback."""
        return "faiss" if self._faiss_index is not None else "numpy"

    def build(self, documents: Sequence[Document]) -> float:
        """Batch-encode documents, normalize vectors, and construct a flat inner-product index."""
        if not documents:
            raise ValueError("at least one document is required to build a dense index")
        ids = [document.document_id for document in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("document IDs must be unique")
        started = time.perf_counter()
        encoding_started = time.perf_counter()
        vectors = np.asarray(self.embedder.encode_documents([doc.retrieval_text for doc in documents]), dtype=np.float32)
        self.encoding_seconds = time.perf_counter() - encoding_started
        if vectors.ndim != 2 or vectors.shape[0] != len(documents) or vectors.shape[1] == 0:
            raise ValueError("embedder must return a nonempty 2D matrix with one vector per document")
        vectors = _normalize(vectors)
        self.documents = list(documents)
        self.embeddings = vectors
        self._faiss_index = _create_faiss_index(vectors)
        self.indexing_seconds = time.perf_counter() - started
        return self.indexing_seconds

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Encode a query once and return the highest cosine-similarity documents."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be greater than 0")
        if not self.documents:
            raise RuntimeError("dense index contains no documents")
        vector = _normalize(np.asarray(self.embedder.encode_query(query), dtype=np.float32).reshape(1, -1))
        return self.search_vector(vector[0], top_k)

    def search_vector(self, query_vector: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        """Rank a supplied query vector without running the embedder again."""
        if top_k < 1:
            raise ValueError("top_k must be greater than 0")
        if not self.documents:
            raise RuntimeError("dense index contains no documents")
        query_vector = _normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        if query_vector.shape[1] != self.embeddings.shape[1]:
            raise ValueError("query embedding dimension does not match the index")
        count = min(top_k, len(self.documents))
        if self._faiss_index is not None:
            scores, positions = self._faiss_index.search(query_vector, count)
            ranked = [(int(position), float(score)) for position, score in zip(positions[0], scores[0]) if position >= 0]
        else:
            scores = self.embeddings @ query_vector[0]
            positions = np.argsort(-scores, kind="stable")[:count]
            ranked = [(int(position), float(scores[position])) for position in positions]
        return [
            SearchResult(
                document_id=self.documents[position].document_id,
                text=self.documents[position].text,
                title=self.documents[position].title,
                score=score,
            )
            for position, score in ranked
        ]

    def save(self, directory: str | Path) -> Path:
        """Persist model metadata, document text, embeddings, and the FAISS index when available."""
        if not self.documents:
            raise RuntimeError("cannot save an empty dense index")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "embeddings.npy", self.embeddings)
        metadata = {
            "model_name": self.model_name,
            "device": self.embedder.device,
            "documents": [
                {"document_id": doc.document_id, "text": doc.text, "title": doc.title}
                for doc in self.documents
            ],
            "backend": self.backend,
        }
        (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if self._faiss_index is not None:
            import faiss

            faiss.write_index(self._faiss_index, str(destination / "index.faiss"))
        return destination

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder) -> "DenseIndex":
        """Restore a saved index only when its stored model name matches the supplied embedder."""
        source = Path(directory)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        if metadata["model_name"] != embedder.model_name:
            raise ValueError(
                f"index model {metadata['model_name']!r} does not match embedder {embedder.model_name!r}"
            )
        instance = cls(embedder)
        instance.documents = [Document(**item) for item in metadata["documents"]]
        instance.embeddings = np.load(source / "embeddings.npy").astype(np.float32, copy=False)
        faiss_path = source / "index.faiss"
        if faiss_path.exists():
            import faiss

            instance._faiss_index = faiss.read_index(str(faiss_path))
        else:
            instance._faiss_index = _create_faiss_index(instance.embeddings)
        return instance


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize a vector or matrix and safely preserve zero vectors."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _create_faiss_index(vectors: np.ndarray):
    """Build a FAISS exact cosine index when FAISS is installed, otherwise return None."""
    if vectors.size == 0:
        return None
    try:
        import faiss
    except ImportError:
        return None
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def model_index_path(root: str | Path, model_name: str) -> Path:
    """Map a public model identifier to a portable directory name under the index root."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._")
    if not safe_name:
        raise ValueError("model name cannot produce an empty index directory")
    return Path(root) / safe_name
