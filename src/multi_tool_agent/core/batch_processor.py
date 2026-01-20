"""Batch query processor with optimization for reduced API calls."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .query_router import QueryRouter, RoutingDecision


@dataclass
class QueryMetrics:
    """Performance metrics for a single query."""

    total_time_ms: float = 0.0
    routing_time_ms: float = 0.0
    policy_time_ms: float = 0.0
    sql_time_ms: float = 0.0
    synthesis_time_ms: float = 0.0
    sql_attempts: int = 0
    pii_masked: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_time_ms": round(self.total_time_ms, 2),
            "routing_time_ms": round(self.routing_time_ms, 2),
            "policy_time_ms": round(self.policy_time_ms, 2),
            "sql_time_ms": round(self.sql_time_ms, 2),
            "synthesis_time_ms": round(self.synthesis_time_ms, 2),
            "sql_attempts": self.sql_attempts,
            "pii_masked": self.pii_masked,
        }


@dataclass
class BatchQueryResult:
    """Result for a single query in a batch."""

    query: str
    response: str
    success: bool
    error: Optional[str] = None
    metrics: QueryMetrics = field(default_factory=QueryMetrics)
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "success": self.success,
            "error": self.error,
            "metrics": self.metrics.to_dict(),
            "request_id": self.request_id,
        }


@dataclass
class BatchMetrics:
    """Aggregate metrics for a batch of queries."""

    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_time_ms: float = 0.0
    avg_time_per_query_ms: float = 0.0
    policy_calls_saved: int = 0
    routing_calls_saved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_per_query_ms": round(self.avg_time_per_query_ms, 2),
            "policy_calls_saved": self.policy_calls_saved,
            "routing_calls_saved": self.routing_calls_saved,
        }


@dataclass
class BatchResult:
    """Complete result for batch processing."""

    results: List[BatchQueryResult]
    batch_metrics: BatchMetrics = field(default_factory=BatchMetrics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "batch_metrics": self.batch_metrics.to_dict(),
        }

    def __iter__(self):
        """Allow iteration over results."""
        return iter(self.results)

    def __len__(self):
        """Return number of results."""
        return len(self.results)


class BatchProcessor:
    """
    Optimizes batch query processing to reduce API calls.

    Optimizations:
    1. Batch routing - route all queries at once to categorize by tool needs
    2. Shared policy context - single RAG call for all policy-related queries
    3. Parallel execution - process independent queries concurrently
    """

    def __init__(
        self,
        router: QueryRouter,
        max_concurrent: int = 5,
    ):
        """
        Initialize batch processor.

        Args:
            router: QueryRouter instance for categorizing queries
            max_concurrent: Maximum concurrent query executions
        """
        self.router = router
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def route_batch(
        self,
        queries: List[str],
    ) -> Tuple[List[RoutingDecision], float]:
        """
        Route all queries and categorize by tool requirements.

        Uses batched routing when possible to reduce API calls.

        Args:
            queries: List of queries to route

        Returns:
            Tuple of (routing decisions, time taken in ms)
        """
        start = time.perf_counter()

        # Route all queries (could be optimized further with batch LLM call)
        decisions = await asyncio.gather(
            *[self.router.route(query) for query in queries],
            return_exceptions=True,
        )

        # Handle any routing failures with fallback
        processed_decisions = []
        for i, decision in enumerate(decisions):
            if isinstance(decision, Exception):
                # Fallback to trying both tools
                processed_decisions.append(
                    RoutingDecision(
                        tools=["policy_accessor", "sql_accessor"],
                        reasoning=f"Fallback routing due to error: {decision}",
                        requires_context_passing=True,
                    )
                )
            else:
                processed_decisions.append(decision)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return processed_decisions, elapsed_ms

    def categorize_queries(
        self,
        queries: List[str],
        decisions: List[RoutingDecision],
    ) -> Dict[str, List[Tuple[int, str, RoutingDecision]]]:
        """
        Categorize queries by their tool requirements.

        Args:
            queries: List of queries
            decisions: Corresponding routing decisions

        Returns:
            Dictionary mapping category to list of (index, query, decision)
        """
        categories: Dict[str, List[Tuple[int, str, RoutingDecision]]] = {
            "policy_only": [],
            "sql_only": [],
            "both": [],
        }

        for i, (query, decision) in enumerate(zip(queries, decisions)):
            tools = set(decision.tools)
            if tools == {"policy_accessor"}:
                categories["policy_only"].append((i, query, decision))
            elif tools == {"sql_accessor"}:
                categories["sql_only"].append((i, query, decision))
            else:
                categories["both"].append((i, query, decision))

        return categories

    def calculate_savings(
        self,
        queries: List[str],
        categories: Dict[str, List[Tuple[int, str, RoutingDecision]]],
    ) -> Tuple[int, int]:
        """
        Calculate API calls saved through batching.

        Args:
            queries: Original query list
            categories: Categorized queries

        Returns:
            Tuple of (policy_calls_saved, routing_calls_saved)
        """
        # Policy calls saved: all policy-needing queries share one context fetch
        policy_needing = len(categories["policy_only"]) + len(categories["both"])
        policy_calls_saved = max(0, policy_needing - 1) if policy_needing > 0 else 0

        # Routing could potentially be batched (future optimization)
        routing_calls_saved = 0

        return policy_calls_saved, routing_calls_saved


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.start_time: float = 0
        self.elapsed_ms: float = 0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000

    @property
    def elapsed(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed_ms
