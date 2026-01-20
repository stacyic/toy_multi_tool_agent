"""RAG tool for policy document retrieval."""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field

from ..interfaces.vector_store import Document
from ..stores.faiss_store import FAISSVectorStore
from .rag_metrics import ChunkInfo, RAGMetricsTracker


@dataclass
class PolicyResult:
    """Result from policy retrieval."""

    content: str
    sources: List[str]
    from_cache: bool = False
    # New metrics fields
    similarity_scores: List[float] = field(default_factory=list)
    latency_ms: float = 0.0


class PolicyAccessor(BaseTool):
    """
    RAG tool for policy document retrieval.

    Features:
    - Loads and chunks policy documents at startup
    - FAISS-based similarity search
    - Caches raw text for graceful degradation
    - Returns relevant policy sections for queries
    """

    name: str = "policy_accessor"
    description: str = (
        "Search company policies for business rules, definitions, and procedures. "
        "Use this tool to find information about return policies, customer tiers (VIP), "
        "shipping, warranties, discounts, and other business policies."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Fields for Pydantic model
    vector_store: FAISSVectorStore = Field(exclude=True)
    raw_policies: str = Field(default="", exclude=True)
    policy_chunks: List[Document] = Field(default_factory=list, exclude=True)
    metrics_tracker: Optional[RAGMetricsTracker] = Field(default=None, exclude=True)
    _initialized: bool = False

    def __init__(
        self,
        policies_path: Optional[Path] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        api_key: Optional[str] = None,
        metrics_tracker: Optional[RAGMetricsTracker] = None,
        **kwargs,
    ):
        """
        Initialize the PolicyAccessor.

        Args:
            policies_path: Path to the policies markdown file
            vector_store: Optional pre-configured vector store
            api_key: OpenAI API key for embeddings
            metrics_tracker: Optional RAGMetricsTracker for retrieval metrics
        """
        # Initialize vector store if not provided
        if vector_store is None:
            vector_store = FAISSVectorStore(api_key=api_key)

        super().__init__(vector_store=vector_store, metrics_tracker=metrics_tracker, **kwargs)

        # Load policies if path provided
        if policies_path:
            self._load_policies(policies_path)

    def _load_policies(self, path: Path) -> None:
        """
        Load and chunk policies from markdown file.

        Args:
            path: Path to policies markdown file
        """
        if not path.exists():
            raise FileNotFoundError(f"Policies file not found: {path}")

        # Read raw content for cache/fallback
        self.raw_policies = path.read_text(encoding="utf-8")

        # Chunk by sections (h2 headers)
        self.policy_chunks = self._chunk_policies(self.raw_policies)

    def _chunk_policies(self, content: str) -> List[Document]:
        """
        Chunk policies by markdown sections with source tracking.

        Strategy:
        - Split on h2 (##) and h3 (###) headers for granular retrieval
        - Each chunk contains the section header and content
        - h3 chunks include parent h2 context for better understanding
        - Preserves context within sections
        - Tracks line numbers for citation

        Args:
            content: Raw markdown content

        Returns:
            List of Document chunks with source location metadata
        """
        chunks = []
        chunk_index = 0

        # Build line number index for the content
        lines = content.split("\n")
        line_offsets = {}  # Maps content position to line number
        current_pos = 0
        for line_num, line in enumerate(lines, 1):
            line_offsets[current_pos] = line_num
            current_pos += len(line) + 1  # +1 for newline

        def find_line_number(text: str, start_search: int = 0) -> int:
            """Find the line number where text starts in the original content."""
            pos = content.find(text, start_search)
            if pos == -1:
                return 1
            # Find the closest line offset
            for offset, line_num in sorted(line_offsets.items()):
                if offset > pos:
                    return max(1, line_num - 1)
            return len(lines)

        # First split by h2 headers
        h2_sections = re.split(r"\n(?=## )", content)
        search_pos = 0

        for h2_section in h2_sections:
            h2_section = h2_section.strip()
            if not h2_section:
                continue

            # Find line number for this h2 section
            h2_start_line = find_line_number(h2_section[:50], search_pos)
            search_pos = content.find(h2_section[:50], search_pos) + len(h2_section)

            # Extract h2 title
            h2_lines = h2_section.split("\n", 1)
            h2_title = h2_lines[0].replace("#", "").strip()

            # Check if this section has h3 subsections
            h3_parts = re.split(r"\n(?=### )", h2_section)

            if len(h3_parts) > 1:
                # Has h3 subsections - chunk each separately
                h2_header = h3_parts[0].strip()  # Content before first h3

                # Add h2 intro as a chunk if it has content beyond the header
                h2_intro_lines = h2_header.split("\n")
                if len(h2_intro_lines) > 1 and h2_intro_lines[1].strip():
                    chunk_index += 1
                    doc = Document(
                        content=h2_header,
                        metadata={
                            "title": h2_title,
                            "source": "policies.md",
                            "type": "policy",
                            "level": "h2",
                            "chunk_index": chunk_index,
                            "line_start": h2_start_line,
                            "line_end": h2_start_line + len(h2_intro_lines) - 1,
                            "section_path": h2_title,
                        },
                    )
                    chunks.append(doc)

                # Add each h3 subsection as a separate chunk
                h3_line_offset = len(h2_header.split("\n"))
                for h3_section in h3_parts[1:]:
                    h3_section = h3_section.strip()
                    if not h3_section:
                        continue

                    # Extract h3 title
                    h3_lines_list = h3_section.split("\n")
                    h3_title = h3_lines_list[0].replace("#", "").strip()
                    h3_start_line = h2_start_line + h3_line_offset

                    # Include parent h2 context in the chunk for better understanding
                    chunk_content = f"## {h2_title}\n\n{h3_section}"

                    chunk_index += 1
                    doc = Document(
                        content=chunk_content,
                        metadata={
                            "title": f"{h2_title} > {h3_title}",
                            "parent_section": h2_title,
                            "source": "policies.md",
                            "type": "policy",
                            "level": "h3",
                            "chunk_index": chunk_index,
                            "line_start": h3_start_line,
                            "line_end": h3_start_line + len(h3_lines_list) - 1,
                            "section_path": f"{h2_title} > {h3_title}",
                        },
                    )
                    chunks.append(doc)
                    h3_line_offset += len(h3_lines_list)
            else:
                # No h3 subsections - keep as single chunk
                chunk_index += 1
                section_lines = h2_section.split("\n")
                doc = Document(
                    content=h2_section,
                    metadata={
                        "title": h2_title,
                        "source": "policies.md",
                        "type": "policy",
                        "level": "h2",
                        "chunk_index": chunk_index,
                        "line_start": h2_start_line,
                        "line_end": h2_start_line + len(section_lines) - 1,
                        "section_path": h2_title,
                    },
                )
                chunks.append(doc)

        return chunks

    async def initialize(self) -> None:
        """
        Initialize the vector store with policy documents.

        Should be called after construction to index documents.
        """
        if self._initialized:
            return

        if self.policy_chunks:
            await self.vector_store.add_documents(self.policy_chunks)
            self._initialized = True

    async def _arun(
        self,
        query: str,
        run_manager: Optional[Any] = None,
    ) -> str:
        """
        Async run the policy search.

        Args:
            query: The search query
            run_manager: Optional callback manager

        Returns:
            Relevant policy content as string
        """
        result = await self.search(query)
        return result.content

    def _run(
        self,
        query: str,
        run_manager: Optional[Any] = None,
    ) -> str:
        """
        Sync run - not recommended, use _arun instead.

        Args:
            query: The search query
            run_manager: Optional callback manager

        Returns:
            Relevant policy content as string
        """
        import asyncio

        return asyncio.run(self._arun(query, run_manager))

    async def search(
        self,
        query: str,
        k: int = 3,
    ) -> PolicyResult:
        """
        Search policies for relevant content.

        Args:
            query: Natural language query
            k: Number of chunks to retrieve

        Returns:
            PolicyResult with content and sources
        """
        start_time = time.perf_counter()

        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        try:
            # Search vector store
            results = await self.vector_store.similarity_search(query, k=k)

            latency_ms = (time.perf_counter() - start_time) * 1000

            if not results:
                return self._fallback_search(query, latency_ms)

            # Combine results - keep sources aligned with scores (no deduplication)
            content_parts = []
            sources = []
            similarity_scores = []
            chunk_details = []

            for result in results:
                content_parts.append(result.document.content)
                similarity_scores.append(result.score)
                metadata = result.document.metadata
                title = metadata.get("title", "Unknown")
                sources.append(title)  # Keep aligned with scores, don't deduplicate

                # Build chunk details for citation
                chunk_details.append(ChunkInfo(
                    source=title,
                    score=result.score,
                    line_start=metadata.get("line_start", 0),
                    line_end=metadata.get("line_end", 0),
                    chunk_index=metadata.get("chunk_index", 0),
                    section_path=metadata.get("section_path", title),
                    level=metadata.get("level", ""),
                ))

            combined_content = "\n\n---\n\n".join(content_parts)

            # Record metrics if tracker is available
            if self.metrics_tracker:
                self.metrics_tracker.record_retrieval(
                    query=query,
                    similarity_scores=similarity_scores,
                    sources=sources,
                    from_fallback=False,
                    latency_ms=latency_ms,
                    chunk_details=chunk_details,
                )

            return PolicyResult(
                content=combined_content,
                sources=sources,
                from_cache=False,
                similarity_scores=similarity_scores,
                latency_ms=latency_ms,
            )

        except Exception:
            latency_ms = (time.perf_counter() - start_time) * 1000
            # Graceful degradation: return cached raw content
            return self._fallback_search(query, latency_ms)

    def _fallback_search(self, query: str, latency_ms: float = 0.0) -> PolicyResult:
        """
        Fallback search using cached raw content.

        Used when vector store fails or returns no results.

        Args:
            query: The search query
            latency_ms: Total latency including vector search attempt

        Returns:
            PolicyResult from cached content
        """
        if not self.raw_policies:
            # Record fallback with no results
            if self.metrics_tracker:
                self.metrics_tracker.record_retrieval(
                    query=query,
                    similarity_scores=[],
                    sources=[],
                    from_fallback=True,
                    latency_ms=latency_ms,
                )
            return PolicyResult(
                content="No policy information available.",
                sources=[],
                from_cache=True,
                similarity_scores=[],
                latency_ms=latency_ms,
            )

        # Simple keyword-based fallback
        query_lower = query.lower()
        relevant_sections = []

        # Split by h2 sections
        sections = re.split(r"\n(?=## )", self.raw_policies)

        for section in sections:
            section_lower = section.lower()
            # Check if any query words appear in section
            if any(word in section_lower for word in query_lower.split()):
                relevant_sections.append(section.strip())

        if relevant_sections:
            content = "\n\n---\n\n".join(relevant_sections[:3])
            sources = ["policies.md (cached)"]
        else:
            # Return all policies if no match
            content = self.raw_policies
            sources = ["policies.md (full cached)"]

        # Record fallback metrics
        if self.metrics_tracker:
            self.metrics_tracker.record_retrieval(
                query=query,
                similarity_scores=[],  # No similarity scores for keyword fallback
                sources=sources,
                from_fallback=True,
                latency_ms=latency_ms,
            )

        return PolicyResult(
            content=content,
            sources=sources,
            from_cache=True,
            similarity_scores=[],  # No similarity scores for keyword fallback
            latency_ms=latency_ms,
        )

    def get_context_for_sql(self, query: str, search_result: PolicyResult) -> str:
        """
        Format policy result as context for SQL generation.

        This method extracts business definitions that might be needed
        for SQL query generation (e.g., VIP customer definition).

        Args:
            query: Original user query
            search_result: Result from policy search

        Returns:
            Formatted context string for SQL generator
        """
        if not search_result.content:
            return ""

        # Format as context
        lines = [
            "Relevant Business Policies:",
            "-" * 30,
            search_result.content,
            "-" * 30,
            "Use these definitions when interpreting the query.",
        ]

        return "\n".join(lines)

    def get_metrics_summary(self) -> Optional[str]:
        """
        Get a summary of RAG retrieval metrics.

        Returns:
            Formatted metrics summary string, or None if no tracker
        """
        if not self.metrics_tracker:
            return None
        return self.metrics_tracker.get_summary()

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get RAG retrieval metrics as a dictionary.

        Returns:
            Metrics dictionary, or None if no tracker
        """
        if not self.metrics_tracker:
            return None
        return self.metrics_tracker.get_metrics().to_dict()

    def export_metrics(self, path: Path) -> None:
        """
        Export RAG metrics to a JSON file.

        Args:
            path: Output path for metrics JSON
        """
        if self.metrics_tracker:
            self.metrics_tracker.export_metrics(path)
