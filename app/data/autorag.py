"""Load the official MTEB AutoRAGRetrieval corpus, queries, and qrels."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.schemas import Document, Query, RetrievalDataset


DEFAULT_DATASET_ID = "mteb/AutoRAGRetrieval"
REQUIRED_CONFIGS = ("corpus", "queries", "qrels")


class DatasetLoadError(RuntimeError):
    """Raised when the optional Hugging Face dataset cannot be loaded safely."""


def _require_datasets() -> Any:
    """Import the optional Hub loader and explain how to install it if absent."""
    try:
        import datasets
    except ImportError as exc:
        raise DatasetLoadError(
            "Loading Hugging Face Parquet datasets requires the optional 'datasets' package. "
            "Install project dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    return datasets


def _rows(dataset: Any) -> list[dict[str, Any]]:
    """Convert Arrow-backed dataset records to plain dictionaries for schema parsing."""
    return [dict(row) for row in dataset]


def _field(row: dict[str, Any], *names: str, required: bool = True) -> Any:
    """Read the first supported MTEB column alias or report a malformed row."""
    for name in names:
        if name in row:
            return row[name]
    if required:
        raise DatasetLoadError(f"Expected one of columns {names}; found {tuple(row)}")
    return None


def load_autorag_retrieval(
    dataset_id: str = DEFAULT_DATASET_ID,
    *,
    split: str = "test",
    revision: str | None = None,
) -> RetrievalDataset:
    """Load a MTEB retrieval dataset and preserve source IDs and qrels scores.

    `datasets` is imported lazily so BM25 and metric unit tests do not require
    model or data-loading dependencies. Data is loaded into memory for the
    initial benchmark-sized dataset.
    """
    if not dataset_id.strip():
        raise ValueError("dataset_id must not be empty")
    if not split.strip():
        raise ValueError("split must not be empty")

    datasets = _require_datasets()
    try:
        loaded: dict[str, list[dict[str, Any]]] = {}
        for config in REQUIRED_CONFIGS:
            kwargs: dict[str, Any] = {"path": dataset_id, "name": config, "split": split}
            if revision is not None:
                kwargs["revision"] = revision
            loaded[config] = _rows(datasets.load_dataset(**kwargs))
    except DatasetLoadError:
        raise
    except Exception as exc:
        raise DatasetLoadError(
            f"Could not load {dataset_id!r} split {split!r}: {exc}"
        ) from exc

    return build_retrieval_dataset(loaded, name=dataset_id, split=split, revision=revision)


def build_retrieval_dataset(
    rows_by_config: dict[str, list[dict[str, Any]]],
    *,
    name: str,
    split: str,
    revision: str | None = None,
) -> RetrievalDataset:
    """Convert MTEB corpus/query/qrels records into validated project schemas."""
    missing_configs = set(REQUIRED_CONFIGS) - set(rows_by_config)
    if missing_configs:
        raise DatasetLoadError(f"Missing MTEB retrieval configs: {sorted(missing_configs)}")

    documents: dict[str, Document] = {}
    for row in rows_by_config["corpus"]:
        document_id = str(_field(row, "id", "_id", "corpus-id"))
        title = str(_field(row, "title", required=False) or "").strip()
        body = str(_field(row, "text", "body", required=False) or "").strip()
        text = body or title
        if document_id in documents:
            raise DatasetLoadError(f"Duplicate corpus document ID: {document_id}")
        try:
            documents[document_id] = Document(document_id=document_id, text=text, title=title)
        except ValueError as exc:
            raise DatasetLoadError(f"Invalid corpus row {document_id!r}: {exc}") from exc

    queries: dict[str, Query] = {}
    for row in rows_by_config["queries"]:
        query_id = str(_field(row, "id", "_id", "query-id"))
        text = str(_field(row, "text", "query") or "").strip()
        if query_id in queries:
            raise DatasetLoadError(f"Duplicate query ID: {query_id}")
        try:
            queries[query_id] = Query(query_id=query_id, text=text)
        except ValueError as exc:
            raise DatasetLoadError(f"Invalid query row {query_id!r}: {exc}") from exc

    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows_by_config["qrels"]:
        query_id = str(_field(row, "query-id", "query_id"))
        document_id = str(_field(row, "corpus-id", "corpus_id", "document_id"))
        score = float(_field(row, "score", "relevance"))
        qrels[query_id][document_id] = max(score, qrels[query_id].get(document_id, float("-inf")))

    try:
        return RetrievalDataset(
            documents=documents,
            queries=queries,
            qrels=dict(qrels),
            name=name,
            metadata={"split": split, "revision": revision or "main"},
        )
    except ValueError as exc:
        raise DatasetLoadError(f"Loaded dataset failed referential validation: {exc}") from exc
