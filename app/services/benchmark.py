"""Quality, latency, and memory measurement for comparable retrieval runs."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Sequence
from typing import Any

from app.evaluation.metrics import evaluate_ranking
from app.schemas import RetrievalDataset
from app.services.search_service import SearchService


class BenchmarkRunner:
    """Run multiple search methods on one fixed dataset and relevance set."""

    def __init__(self, *, embedder: Any = None, index_root: str | None = None) -> None:
        """Store the shared embedder so one model instance serves the full experiment."""
        self.embedder = embedder
        self.index_root = index_root

    def run(
        self,
        dataset: RetrievalDataset,
        *,
        methods: Sequence[str] = ("bm25",),
        k_values: Sequence[int] = (1, 5, 10),
        alpha_values: Sequence[float] = (0.25, 0.5, 0.75),
        latency_repeats: int = 3,
    ) -> dict[str, Any]:
        """Index once, evaluate every requested method, and return reproducibility metadata."""
        if not methods:
            raise ValueError("at least one search method is required")
        if not k_values or any(k < 1 for k in k_values):
            raise ValueError("k_values must contain positive integers")
        if latency_repeats < 1:
            raise ValueError("latency_repeats must be greater than 0")
        if len(set(methods)) != len(methods):
            raise ValueError("methods must not contain duplicates")
        if len(set(k_values)) != len(k_values):
            raise ValueError("k_values must not contain duplicates")
        unknown_methods = set(methods) - {"bm25", "dense", "hybrid"}
        if unknown_methods:
            raise ValueError(f"unsupported search methods: {sorted(unknown_methods)}")
        if ("dense" in methods or "hybrid" in methods) and self.embedder is None:
            raise ValueError("dense and hybrid evaluation require an embedding model")
        if "hybrid" in methods and not alpha_values:
            raise ValueError("hybrid evaluation requires at least one alpha value")
        if any(not 0 <= alpha <= 1 for alpha in alpha_values):
            raise ValueError("all alpha values must be between 0 and 1")
        if len(set(alpha_values)) != len(alpha_values):
            raise ValueError("alpha_values must not contain duplicates")

        memory_before = _memory_snapshot()
        _reset_gpu_peak_memory()
        service = SearchService(embedder=self.embedder, index_root=self.index_root)
        index_times = service.index_documents(list(dataset.documents.values()))
        memory_after_index = _memory_snapshot()
        variants = [(method, None) for method in methods if method != "hybrid"]
        variants.extend(("hybrid", alpha) for alpha in alpha_values if "hybrid" in methods)
        results: list[dict[str, Any]] = []
        max_k = max(k_values)
        for method, alpha in variants:
            ranked_by_query: dict[str, list[str]] = {}
            latencies_ms: list[float] = []
            for query_id, query in dataset.queries.items():
                # Warm each query path once, then measure repeats without changing the evaluation ranking.
                hits = service.search(
                    query.text,
                    method=method,
                    top_k=max_k,
                    alpha=alpha if alpha is not None else 0.5,
                )
                ranked_by_query[query_id] = [hit.document_id for hit in hits]
                for _ in range(latency_repeats):
                    started = time.perf_counter()
                    service.search(
                        query.text,
                        method=method,
                        top_k=max_k,
                        alpha=alpha if alpha is not None else 0.5,
                    )
                    latencies_ms.append((time.perf_counter() - started) * 1000)
            results.append({
                "method": method,
                "alpha": alpha,
                "metrics": evaluate_ranking(ranked_by_query, dataset.qrels, k_values),
                "latency_ms": {
                    "mean": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
                    "median": _percentile(latencies_ms, 0.50),
                    "p95": _percentile(latencies_ms, 0.95),
                },
                "query_count": len(ranked_by_query),
                "latency_measurement_count": len(latencies_ms),
                "indexing_seconds": {
                    "bm25": index_times["bm25_indexing_seconds"],
                    "dense": index_times["dense_indexing_seconds"] if method != "bm25" else None,
                },
                "document_encoding_seconds": (
                    service.dense.encoding_seconds if service.dense is not None and method != "bm25" else None
                ),
            })
        return {
            "experiment_id": str(uuid.uuid4()),
            "dataset": dataset.name,
            "split": dataset.metadata.get("split", "unknown"),
            "revision": dataset.metadata.get("revision", "unknown"),
            "document_count": dataset.document_count,
            "query_count": dataset.query_count,
            "qrels_query_count": len(dataset.qrels),
            "model": getattr(self.embedder, "model_name", None),
            "device": getattr(self.embedder, "device", None),
            "memory": {"before_indexing": memory_before, "after_indexing": memory_after_index},
            "results": results,
        }


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a nearest-rank percentile with a stable result for an empty sequence."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[position]


def _memory_snapshot() -> dict[str, float | str | None]:
    """Collect process RSS and, when PyTorch CUDA is available, allocated GPU memory."""
    try:
        import psutil

        ram_mb: float | None = psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        ram_mb = None
    gpu_name: str | None = None
    gpu_allocated_mb: float | None = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_allocated_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
    except ImportError:
        pass
    gpu_peak_mb: float | None = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_peak_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    except ImportError:
        pass
    return {
        "process_rss_mb": ram_mb,
        "gpu_name": gpu_name,
        "gpu_allocated_mb": gpu_allocated_mb,
        "gpu_peak_allocated_mb": gpu_peak_mb,
    }


def _reset_gpu_peak_memory() -> None:
    """Start a fresh CUDA peak-allocation measurement for the index build and evaluation."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)
    except ImportError:
        return
