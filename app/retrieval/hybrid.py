"""Score-normalized BM25 and dense retrieval combination."""

from __future__ import annotations

from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.schemas import SearchResult


class HybridRetriever:
    """Combine BM25 and dense rankings after independently scaling their scores to [0, 1]."""

    def __init__(self, bm25: BM25Index, dense: DenseIndex, *, candidate_multiplier: int = 3) -> None:
        """Bind both retrieval methods and control how many candidates feed score fusion."""
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be greater than 0")
        self.bm25 = bm25
        self.dense = dense
        self.candidate_multiplier = candidate_multiplier

    def search(self, query: str, top_k: int = 10, *, alpha: float = 0.5) -> list[SearchResult]:
        """Min-max scale each method's candidate scores and rank their union by weighted score."""
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if top_k < 1:
            raise ValueError("top_k must be greater than 0")
        candidate_k = top_k * self.candidate_multiplier
        sparse_results = self.bm25.search(query, candidate_k)
        dense_results = self.dense.search(query, candidate_k)
        sparse_scores = _minmax({result.document_id: result.score for result in sparse_results})
        dense_scores = _minmax({result.document_id: result.score for result in dense_results})
        if alpha == 0:
            candidate_results = sparse_results
        elif alpha == 1:
            candidate_results = dense_results
        else:
            candidate_results = [*sparse_results, *dense_results]
        documents = {result.document_id: result for result in candidate_results}
        combined = {
            document_id: alpha * dense_scores.get(document_id, 0.0)
            + (1.0 - alpha) * sparse_scores.get(document_id, 0.0)
            for document_id in documents
        }
        source_order = {document_id: position for position, document_id in enumerate(documents)}
        ranked_ids = sorted(
            documents,
            key=lambda document_id: (-combined[document_id], source_order[document_id]),
        )[:top_k]
        return [
            SearchResult(
                document_id=document_id,
                text=documents[document_id].text,
                title=documents[document_id].title,
                score=combined[document_id],
            )
            for document_id in ranked_ids
        ]


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    """Scale one result list; a constant list receives equal maximum normalized scores."""
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high == low:
        return {document_id: 1.0 for document_id in scores}
    return {document_id: (score - low) / (high - low) for document_id, score in scores.items()}
