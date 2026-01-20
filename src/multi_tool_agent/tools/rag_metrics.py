"""RAG retrieval metrics tracking for policy accessor.

This module provides simple metrics tracking for RAG retrieval quality,
including similarity scores, fallback rates, and retrieval statistics.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("multi_tool_agent.rag_metrics")


@dataclass
class ChunkInfo:
    """Information about a retrieved chunk for citation."""

    source: str
    score: float
    line_start: int = 0
    line_end: int = 0
    chunk_index: int = 0
    section_path: str = ""
    level: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "score": self.score,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "chunk_index": self.chunk_index,
            "section_path": self.section_path,
            "level": self.level,
        }

    @property
    def citation(self) -> str:
        """Format as a citation string."""
        if self.line_start and self.line_end:
            return f"{self.source} (lines {self.line_start}-{self.line_end})"
        return self.source


@dataclass
class RetrievalResult:
    """Single retrieval result with metrics."""

    query: str
    chunks_retrieved: int
    similarity_scores: List[float]
    sources: List[str]
    from_fallback: bool
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # New: detailed chunk info for citations
    chunk_details: List[ChunkInfo] = field(default_factory=list)

    @property
    def avg_similarity(self) -> float:
        """Average similarity score of retrieved chunks."""
        if not self.similarity_scores:
            return 0.0
        return sum(self.similarity_scores) / len(self.similarity_scores)

    @property
    def max_similarity(self) -> float:
        """Maximum similarity score."""
        return max(self.similarity_scores) if self.similarity_scores else 0.0

    @property
    def min_similarity(self) -> float:
        """Minimum similarity score."""
        return min(self.similarity_scores) if self.similarity_scores else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "chunks_retrieved": self.chunks_retrieved,
            "similarity_scores": self.similarity_scores,
            "avg_similarity": self.avg_similarity,
            "max_similarity": self.max_similarity,
            "min_similarity": self.min_similarity,
            "sources": self.sources,
            "chunk_details": [c.to_dict() for c in self.chunk_details],
            "from_fallback": self.from_fallback,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class RAGMetrics:
    """Aggregated RAG retrieval metrics."""

    total_queries: int = 0
    total_chunks_retrieved: int = 0
    fallback_count: int = 0
    total_latency_ms: float = 0.0
    all_similarity_scores: List[float] = field(default_factory=list)

    # Score distribution buckets
    scores_above_0_8: int = 0  # High relevance
    scores_0_6_to_0_8: int = 0  # Medium relevance
    scores_0_4_to_0_6: int = 0  # Low relevance
    scores_below_0_4: int = 0  # Poor relevance

    @property
    def fallback_rate(self) -> float:
        """Percentage of queries that fell back to keyword search."""
        if self.total_queries == 0:
            return 0.0
        return (self.fallback_count / self.total_queries) * 100

    @property
    def avg_chunks_per_query(self) -> float:
        """Average number of chunks retrieved per query."""
        if self.total_queries == 0:
            return 0.0
        return self.total_chunks_retrieved / self.total_queries

    @property
    def avg_similarity(self) -> float:
        """Average similarity score across all retrievals."""
        if not self.all_similarity_scores:
            return 0.0
        return sum(self.all_similarity_scores) / len(self.all_similarity_scores)

    @property
    def avg_latency_ms(self) -> float:
        """Average retrieval latency in milliseconds."""
        if self.total_queries == 0:
            return 0.0
        return self.total_latency_ms / self.total_queries

    @property
    def high_relevance_rate(self) -> float:
        """Percentage of retrievals with similarity > 0.8."""
        total_scores = len(self.all_similarity_scores)
        if total_scores == 0:
            return 0.0
        return (self.scores_above_0_8 / total_scores) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "total_queries": self.total_queries,
            "total_chunks_retrieved": self.total_chunks_retrieved,
            "avg_chunks_per_query": round(self.avg_chunks_per_query, 2),
            "fallback_count": self.fallback_count,
            "fallback_rate_pct": round(self.fallback_rate, 2),
            "avg_similarity": round(self.avg_similarity, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "score_distribution": {
                "high_relevance_0.8+": self.scores_above_0_8,
                "medium_relevance_0.6-0.8": self.scores_0_6_to_0_8,
                "low_relevance_0.4-0.6": self.scores_0_4_to_0_6,
                "poor_relevance_below_0.4": self.scores_below_0_4,
            },
            "high_relevance_rate_pct": round(self.high_relevance_rate, 2),
        }


class RAGMetricsTracker:
    """
    Tracks RAG retrieval metrics over time.

    Features:
    - Records individual retrieval results
    - Computes aggregate statistics
    - Exports metrics to JSON for analysis
    - Provides summary reports
    """

    def __init__(self, metrics_path: Optional[Path] = None):
        """
        Initialize the metrics tracker.

        Args:
            metrics_path: Optional path to persist metrics (JSON file)
        """
        self.metrics_path = metrics_path
        self.metrics = RAGMetrics()
        self.results: List[RetrievalResult] = []
        self._session_start = datetime.now().isoformat()

    def record_retrieval(
        self,
        query: str,
        similarity_scores: List[float],
        sources: List[str],
        from_fallback: bool,
        latency_ms: float,
        chunk_details: Optional[List[ChunkInfo]] = None,
    ) -> RetrievalResult:
        """
        Record a single retrieval result.

        Args:
            query: The search query
            similarity_scores: Similarity scores for each retrieved chunk
            sources: Source identifiers for retrieved chunks
            from_fallback: Whether fallback search was used
            latency_ms: Retrieval latency in milliseconds
            chunk_details: Optional detailed chunk info for citations

        Returns:
            The recorded RetrievalResult
        """
        result = RetrievalResult(
            query=query,
            chunks_retrieved=len(similarity_scores),
            similarity_scores=similarity_scores,
            sources=sources,
            from_fallback=from_fallback,
            latency_ms=latency_ms,
            chunk_details=chunk_details or [],
        )

        self.results.append(result)
        self._update_metrics(result)

        # Log the retrieval
        logger.info(
            f"[RAG] query='{query[:50]}...' | chunks={result.chunks_retrieved} | "
            f"avg_score={result.avg_similarity:.3f} | fallback={from_fallback} | "
            f"latency={latency_ms:.1f}ms"
        )

        return result

    def _update_metrics(self, result: RetrievalResult) -> None:
        """Update aggregate metrics with a new result."""
        self.metrics.total_queries += 1
        self.metrics.total_chunks_retrieved += result.chunks_retrieved
        self.metrics.total_latency_ms += result.latency_ms

        if result.from_fallback:
            self.metrics.fallback_count += 1

        # Update score distribution
        for score in result.similarity_scores:
            self.metrics.all_similarity_scores.append(score)
            if score >= 0.8:
                self.metrics.scores_above_0_8 += 1
            elif score >= 0.6:
                self.metrics.scores_0_6_to_0_8 += 1
            elif score >= 0.4:
                self.metrics.scores_0_4_to_0_6 += 1
            else:
                self.metrics.scores_below_0_4 += 1

    def get_metrics(self) -> RAGMetrics:
        """Get current aggregate metrics."""
        return self.metrics

    def get_summary(self) -> str:
        """Get a formatted summary of metrics."""
        m = self.metrics
        lines = [
            "RAG Retrieval Metrics Summary",
            "=" * 40,
            f"Total queries: {m.total_queries}",
            f"Total chunks retrieved: {m.total_chunks_retrieved}",
            f"Avg chunks per query: {m.avg_chunks_per_query:.2f}",
            "",
            f"Fallback count: {m.fallback_count}",
            f"Fallback rate: {m.fallback_rate:.1f}%",
            "",
            f"Avg similarity score: {m.avg_similarity:.4f}",
            f"High relevance rate (>0.8): {m.high_relevance_rate:.1f}%",
            "",
            "Score distribution:",
            f"  High (0.8+):    {m.scores_above_0_8}",
            f"  Medium (0.6-0.8): {m.scores_0_6_to_0_8}",
            f"  Low (0.4-0.6):  {m.scores_0_4_to_0_6}",
            f"  Poor (<0.4):    {m.scores_below_0_4}",
            "",
            f"Avg latency: {m.avg_latency_ms:.2f}ms",
        ]
        return "\n".join(lines)

    def get_low_score_queries(self, threshold: float = 0.5) -> List[RetrievalResult]:
        """
        Get queries with average similarity below threshold.

        Args:
            threshold: Minimum acceptable average similarity

        Returns:
            List of low-scoring retrieval results
        """
        return [r for r in self.results if r.avg_similarity < threshold]

    def get_fallback_queries(self) -> List[RetrievalResult]:
        """Get all queries that used fallback search."""
        return [r for r in self.results if r.from_fallback]

    def export_metrics(self, path: Optional[Path] = None) -> None:
        """
        Export metrics to a JSON file.

        Args:
            path: Output path (uses self.metrics_path if not provided)
        """
        output_path = path or self.metrics_path
        if not output_path:
            logger.warning("No metrics path specified, skipping export")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "session_start": self._session_start,
            "export_time": datetime.now().isoformat(),
            "aggregate_metrics": self.metrics.to_dict(),
            "individual_results": [r.to_dict() for r in self.results],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"RAG metrics exported to {output_path}")

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = RAGMetrics()
        self.results = []
        self._session_start = datetime.now().isoformat()


# Global metrics tracker instance (optional singleton pattern)
_global_tracker: Optional[RAGMetricsTracker] = None


def get_metrics_tracker(metrics_path: Optional[Path] = None) -> RAGMetricsTracker:
    """
    Get or create the global metrics tracker.

    Args:
        metrics_path: Optional path for metrics persistence

    Returns:
        The global RAGMetricsTracker instance
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = RAGMetricsTracker(metrics_path)
    return _global_tracker
