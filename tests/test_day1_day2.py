from __future__ import annotations

import unittest

from app.data.autorag import DatasetLoadError, _field, build_retrieval_dataset
from app.evaluation.metrics import evaluate_ranking, mrr_at_k, ndcg_at_k, recall_at_k
from app.retrieval.bm25 import BM25Index
from app.schemas import Document, Query, RetrievalDataset


class SchemaTests(unittest.TestCase):
    def test_rejects_empty_document_and_query(self) -> None:
        """Reject document and query records whose required text is blank."""
        with self.assertRaises(ValueError):
            Document("doc", "  ")
        with self.assertRaises(ValueError):
            Query("query", "  ")

    def test_dataset_preserves_ids_and_checks_references(self) -> None:
        """Preserve valid qrels IDs and reject links to absent documents."""
        dataset = RetrievalDataset(
            documents={"d1": Document("d1", "문서 내용")},
            queries={"q1": Query("q1", "질문")},
            qrels={"q1": {"d1": 1}},
        )
        self.assertEqual(dataset.document_count, 1)
        self.assertEqual(dataset.qrels["q1"]["d1"], 1)
        with self.assertRaises(ValueError):
            RetrievalDataset(
                documents=dataset.documents,
                queries=dataset.queries,
                qrels={"q1": {"missing": 1}},
            )


class BM25Tests(unittest.TestCase):
    def setUp(self) -> None:
        """Build a small Korean corpus shared by the lexical retrieval cases."""
        self.index = BM25Index()
        self.index.index([
            Document("d1", "양자화는 모델 가중치 정밀도를 낮춘다"),
            Document("d2", "벡터 임베딩 검색 시스템"),
            Document("d3", "다른 기술 문서"),
        ])

    def test_search_returns_matching_documents_by_score(self) -> None:
        """Rank a matching Korean document first and keep scores descending."""
        results = self.index.search("양자화는 모델", top_k=5)
        self.assertEqual(results[0].document_id, "d1")
        self.assertEqual([item.score for item in results], sorted(
            (item.score for item in results), reverse=True
        ))
        self.assertEqual(len(results), 1)

    def test_rejects_bad_inputs_and_duplicate_ids(self) -> None:
        """Reject blank queries, invalid cutoffs, and duplicate indexed IDs."""
        with self.assertRaises(ValueError):
            self.index.search(" ")
        with self.assertRaises(ValueError):
            self.index.search("양자화", 0)
        with self.assertRaises(ValueError):
            self.index.index([Document("same", "하나"), Document("same", "둘")])

    def test_unindexed_index_raises(self) -> None:
        """Signal a clear not-ready error when search runs before indexing."""
        with self.assertRaises(RuntimeError):
            BM25Index().search("query")


class MetricTests(unittest.TestCase):
    def test_metric_values_and_cutoff(self) -> None:
        """Check known Recall, MRR, and graded nDCG values at explicit cutoffs."""
        self.assertEqual(recall_at_k(["x", "a"], {"a", "b"}, 2), 0.5)
        self.assertEqual(mrr_at_k(["x", "a"], {"a"}, 2), 0.5)
        self.assertEqual(mrr_at_k(["x", "a"], {"a"}, 1), 0.0)
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 2), 1.0)

    def test_macro_average_includes_unretrieved_queries(self) -> None:
        """Include queries with no retrieved relevant documents as zero scores."""
        scores = evaluate_ranking(
            {"q1": ["a", "b"], "q2": ["x"]},
            {"q1": {"a": 1, "b": 1}, "q2": {"z": 1}},
            (1, 2),
        )
        self.assertEqual(scores["recall@1"], 0.25)
        self.assertEqual(scores["recall@2"], 0.5)
        self.assertEqual(scores["mrr@1"], 0.5)

    def test_rejects_nonpositive_k(self) -> None:
        """Reject a cutoff that cannot describe a ranked prefix."""
        with self.assertRaises(ValueError):
            recall_at_k([], {"d"}, 0)


class LoaderHelperTests(unittest.TestCase):
    def test_column_alias_and_missing_column(self) -> None:
        """Resolve supported MTEB column names and fail when all aliases are absent."""
        self.assertEqual(_field({"query-id": "q1"}, "id", "query-id"), "q1")
        self.assertEqual(_field({"_id": "d1"}, "id", "_id"), "d1")
        with self.assertRaises(DatasetLoadError):
            _field({"other": "value"}, "id")

    def test_builds_mteb_records_and_preserves_grades(self) -> None:
        """Map corpus, query, and qrels fixture rows into the shared data schemas."""
        dataset = build_retrieval_dataset(
            {
                "corpus": [{"_id": "d1", "title": "제목", "text": "본문"}],
                "queries": [{"_id": "q1", "text": "질문"}],
                "qrels": [{"query-id": "q1", "corpus-id": "d1", "score": 2}],
            },
            name="fixture",
            split="test",
        )
        self.assertEqual(dataset.documents["d1"].text, "본문")
        self.assertEqual(dataset.documents["d1"].retrieval_text, "제목\n본문")
        self.assertEqual(dataset.queries["q1"].text, "질문")
        self.assertEqual(dataset.qrels["q1"]["d1"], 2.0)

    def test_build_rejects_missing_mteb_configs(self) -> None:
        """Reject an incomplete dataset bundle before converting any records."""
        with self.assertRaises(DatasetLoadError):
            build_retrieval_dataset({}, name="fixture", split="test")


if __name__ == "__main__":
    unittest.main()
