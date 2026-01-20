"""Unit tests for batch processing functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multi_tool_agent.core.batch_processor import (
    BatchMetrics,
    BatchProcessor,
    BatchQueryResult,
    BatchResult,
    QueryMetrics,
    Timer,
)
from multi_tool_agent.core.query_router import RoutingDecision


class TestQueryMetrics:
    """Tests for QueryMetrics dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        metrics = QueryMetrics()
        assert metrics.total_time_ms == 0.0
        assert metrics.routing_time_ms == 0.0
        assert metrics.policy_time_ms == 0.0
        assert metrics.sql_time_ms == 0.0
        assert metrics.synthesis_time_ms == 0.0
        assert metrics.sql_attempts == 0
        assert metrics.pii_masked is None

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = QueryMetrics(
            total_time_ms=100.5,
            routing_time_ms=10.2,
            sql_time_ms=50.3,
            sql_attempts=2,
            pii_masked={"email": 3},
        )
        result = metrics.to_dict()
        assert result["total_time_ms"] == 100.5
        assert result["sql_attempts"] == 2
        assert result["pii_masked"] == {"email": 3}


class TestBatchQueryResult:
    """Tests for BatchQueryResult dataclass."""

    def test_success_result(self):
        """Test successful query result."""
        result = BatchQueryResult(
            query="What is the return policy?",
            response="The return policy allows returns within 30 days.",
            success=True,
            request_id="abc123",
        )
        assert result.success is True
        assert result.error is None

    def test_failed_result(self):
        """Test failed query result."""
        result = BatchQueryResult(
            query="Invalid query",
            response="Error occurred",
            success=False,
            error="SQL generation failed",
        )
        assert result.success is False
        assert result.error == "SQL generation failed"

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = QueryMetrics(total_time_ms=100)
        result = BatchQueryResult(
            query="Test query",
            response="Test response",
            success=True,
            metrics=metrics,
            request_id="test123",
        )
        d = result.to_dict()
        assert d["query"] == "Test query"
        assert d["success"] is True
        assert d["metrics"]["total_time_ms"] == 100


class TestBatchMetrics:
    """Tests for BatchMetrics dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        metrics = BatchMetrics()
        assert metrics.total_queries == 0
        assert metrics.successful_queries == 0
        assert metrics.failed_queries == 0
        assert metrics.policy_calls_saved == 0

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = BatchMetrics(
            total_queries=10,
            successful_queries=8,
            failed_queries=2,
            total_time_ms=5000,
            avg_time_per_query_ms=500,
            policy_calls_saved=7,
        )
        d = metrics.to_dict()
        assert d["total_queries"] == 10
        assert d["successful_queries"] == 8
        assert d["policy_calls_saved"] == 7


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_iteration(self):
        """Test that BatchResult is iterable."""
        results = [
            BatchQueryResult(query="Q1", response="R1", success=True),
            BatchQueryResult(query="Q2", response="R2", success=True),
        ]
        batch = BatchResult(results=results)

        assert len(batch) == 2
        collected = list(batch)
        assert len(collected) == 2

    def test_to_dict(self):
        """Test dictionary conversion."""
        results = [
            BatchQueryResult(query="Q1", response="R1", success=True),
        ]
        metrics = BatchMetrics(total_queries=1, successful_queries=1)
        batch = BatchResult(results=results, batch_metrics=metrics)

        d = batch.to_dict()
        assert len(d["results"]) == 1
        assert d["batch_metrics"]["total_queries"] == 1


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_basic(self):
        """Test basic timing functionality."""
        with Timer() as t:
            # Simulate some work
            pass
        assert t.elapsed >= 0

    def test_timer_elapsed_property(self):
        """Test elapsed time is in milliseconds."""
        import time

        with Timer() as t:
            time.sleep(0.01)  # 10ms

        assert t.elapsed >= 10  # At least 10ms


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    @pytest.fixture
    def mock_router(self):
        """Create a mock router."""
        router = MagicMock()
        router.route = AsyncMock()
        return router

    @pytest.fixture
    def batch_processor(self, mock_router):
        """Create a BatchProcessor instance."""
        return BatchProcessor(router=mock_router, max_concurrent=3)

    @pytest.mark.asyncio
    async def test_route_batch_success(self, batch_processor, mock_router):
        """Test successful batch routing."""
        mock_router.route.side_effect = [
            RoutingDecision(tools=["policy_accessor"], reasoning="Policy query", requires_context_passing=False),
            RoutingDecision(tools=["sql_accessor"], reasoning="SQL query", requires_context_passing=False),
        ]

        decisions, time_ms = await batch_processor.route_batch(["Q1", "Q2"])

        assert len(decisions) == 2
        assert decisions[0].tools == ["policy_accessor"]
        assert decisions[1].tools == ["sql_accessor"]
        assert time_ms >= 0

    @pytest.mark.asyncio
    async def test_route_batch_with_error(self, batch_processor, mock_router):
        """Test batch routing with error fallback."""
        mock_router.route.side_effect = [
            RoutingDecision(tools=["policy_accessor"], reasoning="OK", requires_context_passing=False),
            Exception("Routing failed"),
        ]

        decisions, _ = await batch_processor.route_batch(["Q1", "Q2"])

        assert len(decisions) == 2
        # Second query should fallback to both tools
        assert set(decisions[1].tools) == {"policy_accessor", "sql_accessor"}

    def test_categorize_queries(self, batch_processor):
        """Test query categorization by tool requirements."""
        queries = ["Q1", "Q2", "Q3"]
        decisions = [
            RoutingDecision(tools=["policy_accessor"], reasoning="", requires_context_passing=False),
            RoutingDecision(tools=["sql_accessor"], reasoning="", requires_context_passing=False),
            RoutingDecision(
                tools=["policy_accessor", "sql_accessor"],
                reasoning="",
                requires_context_passing=True,
            ),
        ]

        categories = batch_processor.categorize_queries(queries, decisions)

        assert len(categories["policy_only"]) == 1
        assert len(categories["sql_only"]) == 1
        assert len(categories["both"]) == 1

    def test_calculate_savings_policy(self, batch_processor):
        """Test API call savings calculation for policy queries."""
        queries = ["Q1", "Q2", "Q3"]
        categories = {
            "policy_only": [(0, "Q1", MagicMock()), (1, "Q2", MagicMock())],
            "sql_only": [],
            "both": [(2, "Q3", MagicMock())],
        }

        policy_saved, routing_saved = batch_processor.calculate_savings(
            queries, categories
        )

        # 3 queries need policy (2 policy_only + 1 both), save 2 calls
        assert policy_saved == 2

    def test_calculate_savings_no_policy(self, batch_processor):
        """Test savings when no policy queries."""
        queries = ["Q1", "Q2"]
        categories = {
            "policy_only": [],
            "sql_only": [(0, "Q1", MagicMock()), (1, "Q2", MagicMock())],
            "both": [],
        }

        policy_saved, routing_saved = batch_processor.calculate_savings(
            queries, categories
        )

        assert policy_saved == 0


class TestBatchProcessorIntegration:
    """Integration tests for batch processing."""

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """Test handling of empty batch."""
        from multi_tool_agent.core.batch_processor import BatchResult

        result = BatchResult(results=[])
        assert len(result) == 0
        assert result.batch_metrics.total_queries == 0

    @pytest.mark.asyncio
    async def test_single_query_batch(self):
        """Test batch with single query."""
        result = BatchQueryResult(
            query="Single query",
            response="Single response",
            success=True,
        )
        batch = BatchResult(
            results=[result],
            batch_metrics=BatchMetrics(total_queries=1, successful_queries=1),
        )

        assert len(batch) == 1
        assert batch.batch_metrics.total_queries == 1
