"""Ranking metrics for retrieval evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _validate_k(k: int) -> None:
    """Raise a clear input error for a zero or negative ranking cutoff."""
    if k <= 0:
        raise ValueError("k must be greater than 0")


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant documents found in the first k results."""
    _validate_k(k)
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def mrr_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant result, or zero when absent."""
    _validate_k(k)
    for rank, document_id in enumerate(ranked_ids[:k], start=1):
        if document_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    """NDCG with graded relevance (gain=2**rel-1, discount=log2(rank+1))."""
    _validate_k(k)

    def dcg(values: Sequence[float]) -> float:
        """Discount graded relevance values by their one-based result positions."""
        return sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(values))

    actual = dcg([max(0.0, float(relevance.get(doc_id, 0.0))) for doc_id in ranked_ids[:k]])
    ideal = dcg(sorted((max(0.0, float(rel)) for rel in relevance.values()), reverse=True)[:k])
    return actual / ideal if ideal else 0.0


def evaluate_ranking(
    ranked_by_query: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    k_values: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Macro-average metrics across qrels queries; missing result lists count as empty."""
    if not qrels:
        raise ValueError("qrels must contain at least one query")
    if not k_values:
        raise ValueError("k_values must contain at least one value")

    output: dict[str, float] = {}
    query_ids = list(qrels)
    for k in k_values:
        _validate_k(k)
        output[f"recall@{k}"] = sum(
            recall_at_k(ranked_by_query.get(query_id, ()),
                        {doc_id for doc_id, rel in qrels[query_id].items() if rel > 0}, k)
            for query_id in query_ids
        ) / len(query_ids)
        output[f"mrr@{k}"] = sum(
            mrr_at_k(ranked_by_query.get(query_id, ()),
                     {doc_id for doc_id, rel in qrels[query_id].items() if rel > 0}, k)
            for query_id in query_ids
        ) / len(query_ids)
        output[f"ndcg@{k}"] = sum(
            ndcg_at_k(ranked_by_query.get(query_id, ()), qrels[query_id], k)
            for query_id in query_ids
        ) / len(query_ids)
    return output
