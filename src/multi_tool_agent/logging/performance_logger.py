"""Enhanced performance logging with component-level timing and debug output."""

import functools
import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class Component(str, Enum):
    """Agent pipeline components for timing."""

    ROUTING = "routing"
    POLICY_SEARCH = "policy_search"
    SQL_GENERATION = "sql_generation"
    SQL_VALIDATION = "sql_validation"
    SQL_EXECUTION = "sql_execution"
    PII_MASKING = "pii_masking"
    RESPONSE_SYNTHESIS = "synthesis"
    BATCH_ORCHESTRATION = "batch_orchestration"


@dataclass
class ComponentTiming:
    """Timing data for a single component execution."""

    component: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark the timing as finished."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "duration_ms": round(self.duration_ms, 2),
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class RequestPerformance:
    """Complete performance data for a single request."""

    request_id: str
    query: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    component_timings: List[ComponentTiming] = field(default_factory=list)
    sql_queries: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None

    @property
    def total_duration_ms(self) -> float:
        """Total request duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000

    @property
    def component_breakdown(self) -> Dict[str, float]:
        """Get timing breakdown by component."""
        breakdown: Dict[str, float] = {}
        for timing in self.component_timings:
            if timing.component in breakdown:
                breakdown[timing.component] += timing.duration_ms
            else:
                breakdown[timing.component] = timing.duration_ms
        return breakdown

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "query": self.query,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "component_breakdown": {
                k: round(v, 2) for k, v in self.component_breakdown.items()
            },
            "component_timings": [t.to_dict() for t in self.component_timings],
            "sql_queries": self.sql_queries,
            "success": self.success,
            "error": self.error,
        }


class PerformanceLogger:
    """
    Enhanced performance logger with component-level timing.

    Features:
    - Per-component timing with context managers
    - SQL query capture with attempt tracking
    - Structured debug output for production
    - JSON export for analysis
    - Visual tree format for console debugging
    """

    def __init__(
        self,
        log_file: Optional[Path] = None,
        level: str = "INFO",
        enable_console: bool = True,
        enable_file: bool = True,
        debug_sql: bool = True,
    ):
        """
        Initialize performance logger.

        Args:
            log_file: Path to performance log file
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            enable_console: Enable console output
            enable_file: Enable file output
            debug_sql: Include SQL queries in debug output
        """
        self.log_file = log_file
        self.debug_sql = debug_sql
        self.current_request: Optional[RequestPerformance] = None
        self._active_timings: List[ComponentTiming] = []

        # Set up logger
        self._logger = logging.getLogger("multi_tool_agent.performance")
        self._logger.setLevel(getattr(logging, level.upper()))
        self._logger.handlers = []
        self._logger.propagate = False  # Prevent duplicate logs to parent logger

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        if enable_file and log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def start_request(self, request_id: str, query: str) -> RequestPerformance:
        """
        Start tracking a new request.

        Args:
            request_id: Unique request identifier
            query: The user's query

        Returns:
            RequestPerformance object
        """
        self.current_request = RequestPerformance(
            request_id=request_id,
            query=query,
        )
        self._active_timings = []
        self._logger.info(f"[PERF] request_id={request_id} | START | query={query[:60]}...")
        return self.current_request

    def end_request(
        self,
        success: bool = True,
        error: Optional[str] = None,
    ) -> Optional[RequestPerformance]:
        """
        End tracking the current request.

        Args:
            success: Whether the request succeeded
            error: Error message if failed

        Returns:
            Completed RequestPerformance object
        """
        if self.current_request is None:
            return None

        self.current_request.end_time = datetime.now()
        self.current_request.success = success
        self.current_request.error = error

        # Log summary
        self._log_request_summary()

        # Write to file
        if self.log_file:
            self._write_perf_json()

        result = self.current_request
        self.current_request = None
        return result

    @contextmanager
    def time_component(
        self,
        component: Component,
        **metadata: Any,
    ) -> Generator[ComponentTiming, None, None]:
        """
        Context manager for timing a component.

        Args:
            component: The component being timed
            **metadata: Additional metadata to record

        Yields:
            ComponentTiming object
        """
        timing = ComponentTiming(
            component=component.value,
            start_time=time.perf_counter(),
            metadata=metadata,
        )
        self._active_timings.append(timing)

        try:
            yield timing
            timing.finish(success=True)
        except Exception as e:
            timing.finish(success=False, error=str(e))
            raise
        finally:
            if self.current_request:
                self.current_request.component_timings.append(timing)
            if timing in self._active_timings:
                self._active_timings.remove(timing)

    def log_sql_attempt(
        self,
        attempt: int,
        sql: str,
        success: bool,
        error: Optional[str] = None,
        execution_time_ms: float = 0.0,
        rows_returned: int = 0,
    ) -> None:
        """
        Log a SQL query attempt.

        Args:
            attempt: Attempt number (1, 2, 3)
            sql: The SQL query
            success: Whether execution succeeded
            error: Error message if failed
            execution_time_ms: Time to execute in ms
            rows_returned: Number of rows returned
        """
        sql_record = {
            "attempt": attempt,
            "sql": sql,
            "success": success,
            "error": error,
            "execution_time_ms": round(execution_time_ms, 2),
            "rows_returned": rows_returned,
            "timestamp": datetime.now().isoformat(),
        }

        if self.current_request:
            self.current_request.sql_queries.append(sql_record)

        # Log to console/file
        status = "OK" if success else "FAILED"
        level = "info" if success else "warning"
        msg = f"[SQL] attempt_{attempt}: {status}"
        if error:
            msg += f" | error={error}"
        if self.debug_sql:
            # Format SQL for readability (single line, truncated)
            sql_display = " ".join(sql.split())[:100]
            if len(sql) > 100:
                sql_display += "..."
            msg += f" | sql={sql_display}"

        getattr(self._logger, level)(msg)

    def log_pii_masked(self, stats: Dict[str, int]) -> None:
        """
        Log PII masking statistics.

        Args:
            stats: Dictionary of PII type -> count
        """
        if not stats:
            return
        total = sum(stats.values())
        details = ", ".join(f"{k}={v}" for k, v in stats.items())
        self._logger.info(f"[PII] masked {total} values ({details})")

    def log_tool_selection(self, tools: List[str], reasoning: str) -> None:
        """
        Log tool selection decision.

        Args:
            tools: Selected tools
            reasoning: Reasoning for selection
        """
        self._logger.info(f"[ROUTING] selected_tools={tools}")
        self._logger.debug(f"[ROUTING] reasoning={reasoning}")

    def _log_request_summary(self) -> None:
        """Log a visual summary of the request performance."""
        if self.current_request is None:
            return

        req = self.current_request
        status = "SUCCESS" if req.success else f"FAILED: {req.error}"

        # Build tree-style output
        lines = [
            f"[PERF] request_id={req.request_id} | total={req.total_duration_ms:.0f}ms | {status}",
        ]

        breakdown = req.component_breakdown
        component_items = list(breakdown.items())

        for i, (component, time_ms) in enumerate(component_items):
            is_last = i == len(component_items) - 1
            prefix = "└─" if is_last else "├─"

            # Find component timing details
            component_timing = next(
                (t for t in req.component_timings if t.component == component),
                None,
            )

            line = f"  {prefix} {component}: {time_ms:.0f}ms"

            # Add details for SQL components
            if component == Component.SQL_GENERATION.value and req.sql_queries:
                attempts = len(req.sql_queries)
                line += f" (attempts: {attempts})"

            if component_timing and not component_timing.success:
                line += f" [FAILED: {component_timing.error}]"

            lines.append(line)

        # Add SQL query details if debug enabled
        if self.debug_sql and req.sql_queries:
            final_sql = next(
                (q for q in reversed(req.sql_queries) if q["success"]),
                req.sql_queries[-1] if req.sql_queries else None,
            )
            if final_sql:
                sql_display = " ".join(final_sql["sql"].split())[:80]
                lines.append(f"[SQL] {sql_display}")

        for line in lines:
            self._logger.info(line)

    def _write_perf_json(self) -> None:
        """Write performance data as JSON."""
        if self.current_request is None or self.log_file is None:
            return

        perf_file = self.log_file.with_suffix(".perf.jsonl")
        with open(perf_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.current_request.to_dict()) + "\n")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current request statistics.

        Returns:
            Dictionary of current request stats
        """
        if self.current_request is None:
            return {}
        return self.current_request.to_dict()


def timed(component: Component) -> Callable[[F], F]:
    """
    Decorator for timing async functions.

    Args:
        component: The component being timed

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Try to find performance logger in args (self.perf_logger)
            perf_logger: Optional[PerformanceLogger] = None
            if args and hasattr(args[0], "perf_logger"):
                perf_logger = args[0].perf_logger

            if perf_logger:
                with perf_logger.time_component(component):
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def timed_sync(component: Component) -> Callable[[F], F]:
    """
    Decorator for timing synchronous functions.

    Args:
        component: The component being timed

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            perf_logger: Optional[PerformanceLogger] = None
            if args and hasattr(args[0], "perf_logger"):
                perf_logger = args[0].perf_logger

            if perf_logger:
                with perf_logger.time_component(component):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator
