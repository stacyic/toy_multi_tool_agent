"""Abstract interfaces for pluggable components."""

from .vector_store import Document, SearchResult, VectorStoreInterface

__all__ = [
    "VectorStoreInterface",
    "Document",
    "SearchResult",
]
