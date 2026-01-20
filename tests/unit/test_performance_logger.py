"""Unit tests for performance logging functionality."""

import tempfile
import time
from datetime import datetime
from pathlib import Path

import pytest

from multi_tool_agent.logging.performance_logger import (
    Component,
    ComponentTiming,
    PerformanceLogger,
    RequestPerformance,
    timed,
    timed_sync,
)


class TestComponent:
    """Tests for Component enum."""

    def test_component_values(self):
        """Test that all components have expected values."""
        assert Component.ROUTING.value == "routing"
        assert Component.POLICY_SEARCH.value == "policy_search"
        assert Component.SQL_GENERATION.value == "sql_generation"
        assert Component.SQL_EXECUTION.value == "sql_execution"
        assert Component.PII_MASKING.value == "pii_masking"
        assert Component.RESPONSE_SYNTHESIS.value == "synthesis"


class TestComponentTiming:
    """Tests for ComponentTiming dataclass."""

    def test_timing_creation(self):
        """Test creating a timing object."""
        timing = ComponentTiming(
            component="routing",
            start_time=time.perf_counter(),
        )
        assert timing.component == "routing"
        assert timing.success is True
        assert timing.error is None

    def test_timing_finish(self):
        """Test finishing a timing."""
        timing = ComponentTiming(
            component="sql_execution",
            start_time=time.perf_counter(),
        )
        time.sleep(0.01)  # 10ms
        timing.finish()

        assert timing.duration_ms >= 10
        assert timing.success is True

    def test_timing_finish_with_error(self):
        """Test finishing a timing with error."""
        timing = ComponentTiming(
            component="sql_execution",
            start_time=time.perf_counter(),
        )
        timing.finish(success=False, error="SQL syntax error")

        assert timing.success is False
        assert timing.error == "SQL syntax error"

    def test_timing_to_dict(self):
        """Test dictionary conversion."""
        timing = ComponentTiming(
            component="routing",
            start_time=time.perf_counter(),
            metadata={"tools": ["sql_accessor"]},
        )
        timing.finish()

        d = timing.to_dict()
        assert d["component"] == "routing"
        assert d["success"] is True
        assert d["metadata"]["tools"] == ["sql_accessor"]


class TestRequestPerformance:
    """Tests for RequestPerformance dataclass."""

    def test_request_creation(self):
        """Test creating a request performance object."""
        req = RequestPerformance(
            request_id="abc123",
            query="What is the return policy?",
        )
        assert req.request_id == "abc123"
        assert req.success is True
        assert len(req.component_timings) == 0

    def test_total_duration(self):
        """Test duration calculation."""
        req = RequestPerformance(
            request_id="test",
            query="test query",
        )
        time.sleep(0.01)
        req.end_time = datetime.now()

        assert req.total_duration_ms >= 10

    def test_component_breakdown(self):
        """Test component timing breakdown."""
        req = RequestPerformance(
            request_id="test",
            query="test query",
        )

        # Add some timings
        timing1 = ComponentTiming(
            component="routing", start_time=time.perf_counter()
        )
        timing1.finish()
        timing1.duration_ms = 50

        timing2 = ComponentTiming(
            component="sql_execution", start_time=time.perf_counter()
        )
        timing2.finish()
        timing2.duration_ms = 100

        req.component_timings = [timing1, timing2]

        breakdown = req.component_breakdown
        assert breakdown["routing"] == 50
        assert breakdown["sql_execution"] == 100

    def test_to_dict(self):
        """Test dictionary conversion."""
        req = RequestPerformance(
            request_id="test123",
            query="test query",
        )
        req.end_time = datetime.now()
        req.sql_queries.append(
            {"attempt": 1, "sql": "SELECT * FROM customers", "success": True}
        )

        d = req.to_dict()
        assert d["request_id"] == "test123"
        assert len(d["sql_queries"]) == 1


class TestPerformanceLogger:
    """Tests for PerformanceLogger class."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def logger(self, temp_log_dir):
        """Create a PerformanceLogger instance."""
        return PerformanceLogger(
            log_file=temp_log_dir / "test.log",
            level="DEBUG",
            enable_console=False,
            enable_file=True,
            debug_sql=True,
        )

    def test_start_request(self, logger):
        """Test starting a request."""
        req = logger.start_request("abc123", "What is the return policy?")

        assert req.request_id == "abc123"
        assert logger.current_request is not None
        assert logger.current_request.request_id == "abc123"

    def test_end_request(self, logger):
        """Test ending a request."""
        logger.start_request("test", "test query")
        result = logger.end_request(success=True)

        assert result is not None
        assert result.success is True
        assert result.end_time is not None

    def test_end_request_with_error(self, logger):
        """Test ending a request with error."""
        logger.start_request("test", "test query")
        result = logger.end_request(success=False, error="Something went wrong")

        assert result.success is False
        assert result.error == "Something went wrong"

    def test_time_component_context_manager(self, logger):
        """Test component timing context manager."""
        logger.start_request("test", "test query")

        with logger.time_component(Component.ROUTING, tools=["sql_accessor"]) as timing:
            time.sleep(0.01)

        assert timing.duration_ms >= 10
        assert timing.metadata["tools"] == ["sql_accessor"]
        assert len(logger.current_request.component_timings) == 1

    def test_time_component_with_exception(self, logger):
        """Test component timing when exception occurs."""
        logger.start_request("test", "test query")

        with pytest.raises(ValueError):
            with logger.time_component(Component.SQL_EXECUTION):
                raise ValueError("Test error")

        # Timing should still be recorded with error
        assert len(logger.current_request.component_timings) == 1
        timing = logger.current_request.component_timings[0]
        assert timing.success is False
        assert "Test error" in timing.error

    def test_log_sql_attempt(self, logger):
        """Test logging SQL attempts."""
        logger.start_request("test", "test query")
        logger.log_sql_attempt(
            attempt=1,
            sql="SELECT * FROM customers WHERE tier = 'VIP'",
            success=True,
            execution_time_ms=45.5,
            rows_returned=10,
        )

        assert len(logger.current_request.sql_queries) == 1
        sql_record = logger.current_request.sql_queries[0]
        assert sql_record["attempt"] == 1
        assert sql_record["success"] is True
        assert sql_record["rows_returned"] == 10

    def test_log_sql_attempt_failure(self, logger):
        """Test logging failed SQL attempt."""
        logger.start_request("test", "test query")
        logger.log_sql_attempt(
            attempt=1,
            sql="SELECT * FORM customers",  # Syntax error
            success=False,
            error="near FORM: syntax error",
        )

        sql_record = logger.current_request.sql_queries[0]
        assert sql_record["success"] is False
        assert "syntax error" in sql_record["error"]

    def test_log_pii_masked(self, logger):
        """Test logging PII masking stats."""
        logger.start_request("test", "test query")
        logger.log_pii_masked({"email": 5, "phone": 3})
        # Should not raise

    def test_log_pii_masked_empty(self, logger):
        """Test logging empty PII stats."""
        logger.start_request("test", "test query")
        logger.log_pii_masked({})
        # Should not raise

    def test_log_tool_selection(self, logger):
        """Test logging tool selection."""
        logger.start_request("test", "test query")
        logger.log_tool_selection(
            tools=["sql_accessor"],
            reasoning="Query appears to require database access",
        )
        # Should not raise

    def test_get_stats(self, logger):
        """Test getting current request stats."""
        # No request started
        stats = logger.get_stats()
        assert stats == {}

        # Start request
        logger.start_request("test", "test query")
        stats = logger.get_stats()
        assert stats["request_id"] == "test"

    def test_write_perf_json(self, logger, temp_log_dir):
        """Test JSON performance file output."""
        logger.start_request("test123", "test query")
        with logger.time_component(Component.ROUTING):
            pass
        logger.end_request(success=True)

        # Check that perf file was created
        perf_file = temp_log_dir / "test.perf.jsonl"
        assert perf_file.exists()

        # Read and verify content
        with open(perf_file, "r") as f:
            import json

            data = json.loads(f.readline())
            assert data["request_id"] == "test123"


class TestTimedDecorators:
    """Tests for timed decorators."""

    def test_timed_decorator_no_logger(self):
        """Test timed decorator without performance logger."""

        class TestClass:
            perf_logger = None

            @timed(Component.ROUTING)
            async def route(self):
                return "result"

        import asyncio

        obj = TestClass()
        result = asyncio.run(obj.route())
        assert result == "result"

    def test_timed_sync_decorator_no_logger(self):
        """Test timed_sync decorator without performance logger."""

        class TestClass:
            perf_logger = None

            @timed_sync(Component.SQL_VALIDATION)
            def validate(self):
                return True

        obj = TestClass()
        result = obj.validate()
        assert result is True


class TestPerformanceLoggerIntegration:
    """Integration tests for performance logging."""

    def test_full_request_flow(self):
        """Test a complete request flow."""
        logger = PerformanceLogger(
            log_file=None,
            enable_console=False,
            enable_file=False,
        )

        # Start request
        logger.start_request("int-test", "How many VIP customers?")

        # Time routing
        with logger.time_component(Component.ROUTING, tools=["sql_accessor"]):
            time.sleep(0.005)

        # Time SQL execution
        with logger.time_component(Component.SQL_EXECUTION):
            time.sleep(0.010)
            logger.log_sql_attempt(
                attempt=1,
                sql="SELECT COUNT(*) FROM customers WHERE tier = 'VIP'",
                success=True,
                execution_time_ms=10,
                rows_returned=1,
            )

        # Log PII
        logger.log_pii_masked({"email": 0})

        # Time synthesis
        with logger.time_component(Component.RESPONSE_SYNTHESIS):
            time.sleep(0.005)

        # End request
        result = logger.end_request(success=True)

        # Verify
        assert result.success is True
        assert len(result.component_timings) == 3
        assert len(result.sql_queries) == 1
        assert result.total_duration_ms >= 20  # At least 20ms total
