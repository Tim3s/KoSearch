from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np
import httpx

from app.models.embedder import Embedder
from app.retrieval.dense import DenseIndex, model_index_path
from app.retrieval.hybrid import HybridRetriever
from app.schemas import Document
from app.services.experiment_service import ExperimentStore
from app.services.search_service import SearchNotReady, SearchService


class FakeEmbedder:
    """Provide deterministic small vectors so dense tests need no model download."""

    model_name = "fake/test-embedder"
    device = "cpu"

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Map fixture text to vectors with predictable cosine ranks."""
        return np.asarray([[1, 0] if "fruit" in text else [0, 1] for text in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Map fixture query terms into the same two-dimensional vector space."""
        return np.asarray([1, 0] if "fruit" in text else [0, 1], dtype=np.float32)


class DenseHybridTests(unittest.TestCase):
    """Verify dense, hybrid, persistence, and service behavior without neural weights."""

    def setUp(self) -> None:
        """Build a tiny deterministic retrieval collection for each test."""
        self.documents = [
            Document("d-fruit", "fruit apple", "Fruit"),
            Document("d-car", "vehicle car", "Vehicle"),
        ]
        self.embedder = FakeEmbedder()

    def test_dense_cosine_search_and_model_checked_persistence(self) -> None:
        """Return the nearest vector and restore its model-specific index from disk."""
        index = DenseIndex(self.embedder)
        index.build(self.documents)
        self.assertEqual(index.search("fruit query", 1)[0].document_id, "d-fruit")
        with tempfile.TemporaryDirectory() as folder:
            index.save(folder)
            restored = DenseIndex.load(folder, self.embedder)
            self.assertEqual(restored.search("car query", 1)[0].document_id, "d-car")
            wrong_model = FakeEmbedder()
            wrong_model.model_name = "another-model"
            with self.assertRaises(ValueError):
                DenseIndex.load(folder, wrong_model)

    def test_hybrid_normalizes_and_returns_common_results(self) -> None:
        """Combine BM25 and dense rankings into one score-sorted result contract."""
        service = SearchService(embedder=self.embedder)
        service.index_documents(self.documents)
        results = service.search("fruit", method="hybrid", top_k=2, alpha=0.5)
        self.assertEqual(results[0].document_id, "d-fruit")
        self.assertTrue(all(0.0 <= result.score <= 1.0 for result in results))
        with self.assertRaises(ValueError):
            HybridRetriever(service.bm25, service.dense).search("fruit", alpha=1.1)

    def test_hybrid_weight_endpoints_use_only_the_weighted_retriever(self) -> None:
        """Keep alpha zero and one rankings within only the method with nonzero weight."""
        service = SearchService(embedder=self.embedder)
        service.index_documents(self.documents)
        hybrid = HybridRetriever(service.bm25, service.dense)
        bm25_ids = [result.document_id for result in service.bm25.search("fruit", 2)]
        dense_ids = [result.document_id for result in service.dense.search("fruit", 2)]
        self.assertEqual([result.document_id for result in hybrid.search("fruit", 2, alpha=0)], bm25_ids)
        self.assertEqual([result.document_id for result in hybrid.search("fruit", 2, alpha=1)], dense_ids)

    def test_search_service_requires_index_and_dense_model(self) -> None:
        """Return explicit not-ready errors before indexing and for absent dense models."""
        service = SearchService()
        with self.assertRaises(SearchNotReady):
            service.search("fruit")
        service.index_documents(self.documents)
        with self.assertRaises(SearchNotReady):
            service.search("fruit", method="dense")

    def test_model_index_path_is_safe(self) -> None:
        """Convert a model identifier into a non-traversing directory name."""
        result = model_index_path("indexes", "org/model-name")
        self.assertEqual(result, Path("indexes") / "org_model-name")

    def test_experiment_json_csv_storage_and_path_validation(self) -> None:
        """Write a report in both formats and reject path-like retrieval identifiers."""
        report = {
            "experiment_id": "exp_123",
            "dataset": "fixture",
            "split": "test",
            "revision": "sha",
            "model": None,
            "results": [{
                "method": "bm25",
                "alpha": None,
                "query_count": 1,
                "latency_ms": {"mean": 1.0, "median": 1.0, "p95": 1.0},
                "indexing_seconds": {"bm25": 0.01, "dense": None},
                "metrics": {"recall@1": 1.0},
            }],
        }
        with tempfile.TemporaryDirectory() as folder:
            store = ExperimentStore(folder)
            paths = store.save(report)
            self.assertEqual(store.get("exp_123"), report)
            self.assertTrue(Path(paths["csv"]).is_file())
            with self.assertRaises(ValueError):
                store.get("../outside")


class ApiTests(unittest.TestCase):
    """Exercise HTTP schemas and the in-memory BM25 API without optional models."""

    def test_document_index_search_health_and_validation(self) -> None:
        """Index documents over HTTP, search them, and check errors for invalid states."""
        from app.main import create_app

        async def exercise_requests() -> None:
            """Manage FastAPI lifespan and issue in-process ASGI requests without a server."""
            application = create_app()
            async with application.router.lifespan_context(application):
                transport = httpx.ASGITransport(app=application)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    self.assertEqual((await client.get("/health")).status_code, 200)
                    self.assertEqual((await client.post("/search", json={"query": "fruit"})).status_code, 503)
                    bad_index = await client.post("/documents/index", json={"documents": [
                        {"document_id": "same", "text": "one"},
                        {"document_id": "same", "text": "two"},
                    ]})
                    self.assertEqual(bad_index.status_code, 422)
                    indexed = await client.post("/documents/index", json={"documents": [
                        {"document_id": "d1", "text": "fruit apple"},
                        {"document_id": "d2", "text": "vehicle car"},
                    ]})
                    self.assertEqual(indexed.status_code, 200)
                    found = await client.post("/search", json={"query": "fruit", "method": "bm25"})
                    self.assertEqual(found.status_code, 200)
                    self.assertEqual(found.json()["results"][0]["document_id"], "d1")
                    self.assertEqual((await client.get("/experiments/unknown")).status_code, 404)

        asyncio.run(exercise_requests())


if __name__ == "__main__":
    unittest.main()
