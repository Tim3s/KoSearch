"""Validated request and response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


SearchMethod = Literal["bm25", "dense", "hybrid"]


class DocumentInput(BaseModel):
    """Represent one nonempty document supplied for indexing."""

    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    title: str = ""

    @field_validator("document_id", "text")
    @classmethod
    def validate_not_blank(cls, value: str) -> str:
        """Reject whitespace-only required values after Pydantic's length check."""
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class IndexDocumentsRequest(BaseModel):
    """Wrap a nonempty unique document collection for index replacement."""

    documents: list[DocumentInput] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "IndexDocumentsRequest":
        """Reject duplicate IDs before replacing the current in-memory collection."""
        ids = [document.document_id for document in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("document_id values must be unique")
        return self


class SearchRequest(BaseModel):
    """Validate search query, retrieval method, top-k, and hybrid weight."""

    query: str = Field(min_length=1)
    method: SearchMethod = "bm25"
    top_k: int = Field(default=5, ge=1, le=100)
    alpha: float = Field(default=0.5, ge=0, le=1)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Reject a query that contains only whitespace."""
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class SearchHit(BaseModel):
    """Return one ranked document and its search score."""

    document_id: str
    title: str
    text: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    """Return a search result list with method and measured request latency."""

    query: str
    method: SearchMethod
    latency_ms: float
    results: list[SearchHit]


class EvaluateRequest(BaseModel):
    """Restrict evaluation to a registered benchmark and supported methods."""

    dataset: Literal["autorag"] = "autorag"
    methods: list[SearchMethod] = Field(default_factory=lambda: ["bm25"])
    k_values: list[int] = Field(default_factory=lambda: [1, 5, 10])
    alpha_values: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75])

    @model_validator(mode="after")
    def validate_evaluation_parameters(self) -> "EvaluateRequest":
        """Require nonempty unique positive K values and bounded hybrid weights."""
        if not self.methods:
            raise ValueError("methods must contain at least one method")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must not contain duplicates")
        if not self.k_values or any(value < 1 for value in self.k_values):
            raise ValueError("k_values must contain positive integers")
        if len(set(self.k_values)) != len(self.k_values):
            raise ValueError("k_values must be unique")
        if not self.alpha_values or any(not 0 <= value <= 1 for value in self.alpha_values):
            raise ValueError("alpha_values must be nonempty values between 0 and 1")
        if len(set(self.alpha_values)) != len(self.alpha_values):
            raise ValueError("alpha_values must not contain duplicates")
        return self
