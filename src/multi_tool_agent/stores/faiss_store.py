"""FAISS vector store implementation."""

import asyncio
import uuid
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings

from ..interfaces.vector_store import Document, SearchResult, VectorStoreInterface


class FAISSVectorStore(VectorStoreInterface):
    """
    FAISS-based vector store implementation.

    Features:
    - In-memory storage with FAISS for fast similarity search
    - OpenAI embeddings integration
    - Async interface with sync FAISS operations
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimension: int = 1536,
    ):
        """
        Initialize the FAISS vector store.

        Args:
            embedding_model: OpenAI embedding model name
            api_key: OpenAI API key (optional, uses env var if not provided)
            dimension: Embedding dimension (1536 for text-embedding-3-small)
        """
        self.embedding_model = embedding_model
        self.dimension = dimension

        # Initialize embeddings
        kwargs = {"model": embedding_model}
        if api_key:
            kwargs["api_key"] = api_key
        self._embeddings = OpenAIEmbeddings(**kwargs)

        # Initialize FAISS index
        self._index = faiss.IndexFlatL2(dimension)

        # Document storage (id -> Document)
        self._documents: Dict[str, Document] = {}

        # Mapping from FAISS index position to document ID
        self._index_to_id: List[str] = []

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._embeddings.embed_documents, texts
        )

    async def _get_query_embedding(self, text: str) -> List[float]:
        """Get embedding for a single query text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embeddings.embed_query, text)

    async def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add documents to the store.

        Args:
            documents: List of Document objects to add
            embeddings: Optional pre-computed embeddings

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        # Generate IDs for documents without them
        for doc in documents:
            if doc.id is None:
                doc.id = str(uuid.uuid4())

        # Get embeddings if not provided
        if embeddings is None:
            texts = [doc.content for doc in documents]
            embeddings = await self._get_embeddings(texts)

        # Convert to numpy array and add to FAISS
        vectors = np.array(embeddings, dtype=np.float32)
        self._index.add(vectors)

        # Store documents and update mapping
        ids = []
        for doc in documents:
            self._documents[doc.id] = doc
            self._index_to_id.append(doc.id)
            ids.append(doc.id)

        return ids

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
            filter: Optional metadata filter (not fully implemented for FAISS)

        Returns:
            List of SearchResult objects
        """
        # Get query embedding
        query_embedding = await self._get_query_embedding(query)

        return await self.similarity_search_by_vector(query_embedding, k, filter)

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
            List of SearchResult objects
        """
        if self._index.ntotal == 0:
            return []

        # Adjust k if we have fewer documents
        k = min(k, self._index.ntotal)

        # Convert to numpy array
        query_vector = np.array([embedding], dtype=np.float32)

        # Search FAISS index
        distances, indices = self._index.search(query_vector, k)

        # Build results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self._index_to_id):
                continue

            doc_id = self._index_to_id[idx]
            doc = self._documents.get(doc_id)

            if doc is None:
                continue

            # Apply metadata filter if provided
            if filter:
                match = all(
                    doc.metadata.get(key) == value for key, value in filter.items()
                )
                if not match:
                    continue

            # Convert L2 distance to similarity score (inverse)
            # Lower distance = higher similarity
            score = 1.0 / (1.0 + distances[0][i])

            results.append(SearchResult(document=doc, score=score))

        return results

    async def delete(self, ids: List[str]) -> bool:
        """
        Delete documents by ID.

        Note: FAISS doesn't support direct deletion, so we rebuild the index.

        Args:
            ids: List of document IDs to delete

        Returns:
            True if deletion was successful
        """
        ids_to_delete = set(ids)

        # Remove from document storage
        for doc_id in ids_to_delete:
            self._documents.pop(doc_id, None)

        # Rebuild index without deleted documents
        remaining_docs = list(self._documents.values())

        if remaining_docs:
            texts = [doc.content for doc in remaining_docs]
            embeddings = await self._get_embeddings(texts)

            # Reset index
            self._index = faiss.IndexFlatL2(self.dimension)
            vectors = np.array(embeddings, dtype=np.float32)
            self._index.add(vectors)

            # Reset mapping
            self._index_to_id = [doc.id for doc in remaining_docs]
        else:
            # Empty index
            self._index = faiss.IndexFlatL2(self.dimension)
            self._index_to_id = []

        return True

    async def clear(self) -> bool:
        """Clear all documents from the store."""
        self._index = faiss.IndexFlatL2(self.dimension)
        self._documents = {}
        self._index_to_id = []
        return True

    @property
    def count(self) -> int:
        """Return the number of documents in the store."""
        return len(self._documents)
