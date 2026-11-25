"""Small, dependency-free BM25 implementation for Korean text."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

from app.schemas import Document, SearchResult


_TOKEN_PATTERN = re.compile(r"[가-힣]+|[A-Za-z]+|\d+")


def tokenize(text: str) -> list[str]:
    """Extract Korean words, Latin words, and numbers without external models."""
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


class BM25Index:
    """BM25 Okapi index. Rebuilding replaces the complete indexed collection."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        """Configure BM25 Okapi saturation and document-length normalization values."""
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self._documents: list[Document] = []
        self._term_frequencies: list[Counter[str]] = []
        self._postings: dict[str, list[int]] = defaultdict(list)
        self._document_frequency: Counter[str] = Counter()
        self._lengths: list[int] = []
        self._average_length = 0.0

    @property
    def document_count(self) -> int:
        """Return how many documents are currently present in the index."""
        return len(self._documents)

    def index(self, documents: Iterable[Document]) -> None:
        """Replace the index; duplicate IDs and blank documents are rejected."""
        new_documents = list(documents)
        seen: set[str] = set()
        for document in new_documents:
            if document.document_id in seen:
                raise ValueError(f"duplicate document_id: {document.document_id}")
            seen.add(document.document_id)

        self._documents = new_documents
        self._term_frequencies = [Counter(tokenize(doc.retrieval_text)) for doc in new_documents]
        self._lengths = [sum(freq.values()) for freq in self._term_frequencies]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._postings = defaultdict(list)
        self._document_frequency = Counter()
        for index, frequencies in enumerate(self._term_frequencies):
            for term in frequencies:
                self._postings[term].append(index)
                self._document_frequency[term] += 1

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Score query terms with BM25 and return positive-match results in stable order."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not self._documents:
            raise RuntimeError("index contains no documents")

        query_terms = tokenize(query)
        scores: Counter[int] = Counter()
        n_documents = len(self._documents)
        for term in query_terms:
            document_frequency = self._document_frequency.get(term, 0)
            if not document_frequency:
                continue
            # BM25 Okapi IDF; this form remains positive for common terms.
            inverse_document_frequency = math.log(
                1 + (n_documents - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for index in self._postings[term]:
                term_frequency = self._term_frequencies[index][term]
                length_norm = 1 - self.b + self.b * self._lengths[index] / self._average_length
                scores[index] += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1)
                    / (term_frequency + self.k1 * length_norm)
                )

        ranked_indices = sorted(scores, key=lambda index: (-scores[index], index))[:top_k]
        return [
            SearchResult(
                document_id=self._documents[index].document_id,
                text=self._documents[index].text,
                score=float(scores[index]),
                title=self._documents[index].title,
            )
            for index in ranked_indices
        ]
