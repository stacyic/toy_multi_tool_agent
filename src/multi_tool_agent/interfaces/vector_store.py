"""Abstract interface for vector stores.

This interface allows swapping vector store implementations
(FAISS, Chroma, Pinecone, etc.) without changing the rest of the codebase.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """Represents a document chunk with content and metadata."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None


@dataclass
class SearchResult:
    """Result from a similarity search."""

    document: Document
    score: float


class VectorStoreInterface(ABC):
    """
    Abstract interface for vector stores.

    Implementations must provide methods for:
    - Adding documents with embeddings
    - Similarity search
    - Deleting documents

    Example implementations:
    - FAISSVectorStore (in-memory, fast)
    - ChromaVectorStore (persistent, feature-rich)
    - PineconeVectorStore (cloud-hosted, scalable)
    """

    @abstractmethod
    async def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add documents to the store.

        Args:
            documents: List of Document objects to add
            embeddings: Optional pre-computed embeddings. If not provided,
                       embeddings will be generated.

        Returns:
            List of document IDs for the added documents
        """
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for documents similar to the query.

        Args:
            query: The search query text
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of SearchResult objects with documents and scores
        """
        pass

    @abstractmethod
    async def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for documents using a pre-computed embedding vector.

        Args:
            embedding: The query embedding vector
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of SearchResult objects with documents and scores
        """
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> bool:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """
        Clear all documents from the store.

        Returns:
            True if clearing was successful
        """
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the store."""
        pass
