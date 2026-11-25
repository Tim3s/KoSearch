"""Validated in-memory schemas shared by loaders and retrieval components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    text: str
    title: str = ""

    def __post_init__(self) -> None:
        """Reject documents that cannot be identified or searched as text."""
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")
        if not self.text.strip():
            raise ValueError(f"document text must not be empty: {self.document_id}")

    @property
    def retrieval_text(self) -> str:
        """Return title plus body so both fields can contribute to lexical and dense search."""
        return "\n".join(part for part in (self.title.strip(), self.text.strip()) if part)


@dataclass(frozen=True, slots=True)
class Query:
    query_id: str
    text: str

    def __post_init__(self) -> None:
        """Reject queries missing their stable ID or query text."""
        if not self.query_id.strip():
            raise ValueError("query_id must not be empty")
        if not self.text.strip():
            raise ValueError(f"query text must not be empty: {self.query_id}")


@dataclass(frozen=True, slots=True)
class SearchResult:
    document_id: str
    text: str
    score: float
    title: str = ""


@dataclass(frozen=True, slots=True)
class RetrievalDataset:
    """Documents, queries, and graded qrels keyed by stable benchmark IDs."""

    documents: Mapping[str, Document]
    queries: Mapping[str, Query]
    qrels: Mapping[str, Mapping[str, float]]
    name: str = "custom"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure the corpus, query set, and every qrels reference agree."""
        if not self.documents:
            raise ValueError("dataset must contain at least one document")
        if not self.queries:
            raise ValueError("dataset must contain at least one query")
        missing_queries = set(self.qrels) - set(self.queries)
        if missing_queries:
            raise ValueError(f"qrels reference unknown queries: {sorted(missing_queries)[:5]}")
        missing_documents = {
            document_id
            for judgments in self.qrels.values()
            for document_id in judgments
            if document_id not in self.documents
        }
        if missing_documents:
            raise ValueError(f"qrels reference unknown documents: {sorted(missing_documents)[:5]}")

    @property
    def document_count(self) -> int:
        """Return corpus size without exposing a separate mutable counter."""
        return len(self.documents)

    @property
    def query_count(self) -> int:
        """Return the number of loaded query records."""
        return len(self.queries)
