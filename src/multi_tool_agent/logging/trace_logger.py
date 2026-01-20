"""Configurable trace logging with cost tracking."""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# OpenAI pricing per 1K tokens (as of 2024)
OPENAI_PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
}


@dataclass
class TokenUsage:
    """Token usage for a single API call."""

    model: str
    input_tokens: int
    output_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def cost(self) -> float:
        """Calculate cost for this usage."""
        pricing = OPENAI_PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        input_cost = (self.input_tokens / 1000) * pricing["input"]
        output_cost = (self.output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost


@dataclass
class TraceEvent:
    """A single trace event."""

    event_type: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "level": self.level,
        }


@dataclass
class RequestTrace:
    """Trace for a single request."""

    request_id: str
    query: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    events: List[TraceEvent] = field(default_factory=list)
    token_usage: List[TokenUsage] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Calculate request duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    @property
    def total_cost(self) -> float:
        """Calculate total cost for this request."""
        return sum(usage.cost for usage in self.token_usage)

    @property
    def total_tokens(self) -> Dict[str, int]:
        """Calculate total tokens used."""
        return {
            "input": sum(u.input_tokens for u in self.token_usage),
            "output": sum(u.output_tokens for u in self.token_usage),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "request_id": self.request_id,
            "query": self.query,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "events": [e.to_dict() for e in self.events],
            "token_usage": {
                "total_input": self.total_tokens["input"],
                "total_output": self.total_tokens["output"],
            },
            "total_cost_usd": self.total_cost,
            "success": self.success,
            "error": self.error,
        }


class TraceLogger:
    """
    Configurable trace logger with cost tracking.

    Features:
    - Logs to console, file, or both
    - Tracks token usage and costs per request
    - Structured events for tool selection, SQL attempts, retries
    - JSON-formatted file logs for analysis
    """

    def __init__(
        self,
        destination: Literal["console", "file", "both"] = "both",
        log_file: Optional[Path] = None,
        level: str = "INFO",
        include_costs: bool = True,
    ):
        """
        Initialize the trace logger.

        Args:
            destination: Where to log (console, file, or both)
            log_file: Path to log file (required if destination includes file)
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            include_costs: Whether to include cost tracking
        """
        self.destination = destination
        self.log_file = log_file
        self.include_costs = include_costs
        self.current_trace: Optional[RequestTrace] = None

        # Set up Python logger
        self._logger = logging.getLogger("multi_tool_agent")
        self._logger.setLevel(getattr(logging, level.upper()))
        self._logger.handlers = []  # Clear existing handlers

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        if destination in ("console", "both"):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        if destination in ("file", "both") and log_file:
            # Ensure log directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def start_request(self, request_id: str, query: str) -> RequestTrace:
        """
        Start tracing a new request.

        Args:
            request_id: Unique request identifier
            query: The user's query

        Returns:
            New RequestTrace object
        """
        self.current_trace = RequestTrace(
            request_id=request_id,
            query=query,
        )
        self._log_event("request_start", f"Request started: {query[:50]}...")
        return self.current_trace

    def end_request(self, success: bool = True, error: Optional[str] = None) -> None:
        """
        End the current request trace.

        Args:
            success: Whether the request succeeded
            error: Error message if failed
        """
        if self.current_trace is None:
            return

        self.current_trace.end_time = datetime.now()
        self.current_trace.success = success
        self.current_trace.error = error

        # Log summary
        summary_parts = [
            f"Request completed in {self.current_trace.duration_ms:.0f}ms",
            f"Success: {success}",
        ]

        if self.include_costs:
            tokens = self.current_trace.total_tokens
            summary_parts.append(
                f"Tokens: {tokens['input']} in / {tokens['output']} out"
            )
            summary_parts.append(f"Cost: ${self.current_trace.total_cost:.4f}")

        if error:
            summary_parts.append(f"Error: {error}")

        self._log_event("request_end", " | ".join(summary_parts))

        # Write full trace to file if configured
        if self.destination in ("file", "both") and self.log_file:
            self._write_trace_json()

    def log_tool_selection(self, tools: List[str], reasoning: str) -> None:
        """
        Log tool selection decision.

        Args:
            tools: List of selected tools
            reasoning: Reasoning for selection
        """
        self._log_event(
            "tool_selection",
            f"Selected tools: {tools}",
            metadata={"tools": tools, "reasoning": reasoning},
        )

    def log_sql_attempt(
        self,
        attempt: int,
        sql: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """
        Log a SQL execution attempt.

        Args:
            attempt: Attempt number (1-3)
            sql: The SQL query
            success: Whether execution succeeded
            error: Error message if failed
        """
        level = "INFO" if success else "WARNING"
        # Format SQL for logging (single line)
        sql_oneline = " ".join(sql.split()) if sql else "(empty)"
        message = f"SQL attempt {attempt}: {'Success' if success else 'Failed'}"
        if error:
            message += f" - {error}"
        message += f" | Query: {sql_oneline}"

        self._log_event(
            "sql_attempt",
            message,
            metadata={"attempt": attempt, "sql": sql, "success": success, "error": error},
            level=level,
        )

    def log_sql_validation(
        self,
        is_valid: bool,
        errors: List[str],
        sql: Optional[str] = None,
    ) -> None:
        """
        Log SQL validation result.

        Args:
            is_valid: Whether SQL passed validation
            errors: List of validation errors
            sql: The SQL query that was validated
        """
        level = "DEBUG" if is_valid else "WARNING"
        message = "SQL validation: " + ("Passed" if is_valid else f"Failed - {errors}")
        if sql:
            sql_oneline = " ".join(sql.split())
            message += f" | Query: {sql_oneline}"

        self._log_event(
            "sql_validation",
            message,
            metadata={"is_valid": is_valid, "errors": errors},
            level=level,
        )

    def log_pii_masked(self, stats: Dict[str, int]) -> None:
        """
        Log PII masking statistics.

        Args:
            stats: Dictionary of PII type -> count masked
        """
        if not stats:
            return

        total = sum(stats.values())
        details = ", ".join(f"{k}: {v}" for k, v in stats.items())
        self._log_event(
            "pii_masked",
            f"Masked {total} PII values ({details})",
            metadata=stats,
        )

    def log_policy_search(
        self,
        query: str,
        results_count: int,
        from_cache: bool,
    ) -> None:
        """
        Log policy search.

        Args:
            query: Search query
            results_count: Number of results found
            from_cache: Whether results came from cache
        """
        source = "cache" if from_cache else "vector store"
        self._log_event(
            "policy_search",
            f"Policy search: {results_count} results from {source}",
            metadata={"query": query, "results": results_count, "from_cache": from_cache},
        )

    def log_token_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """
        Log token usage for cost tracking.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        if self.current_trace is None:
            return

        usage = TokenUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.current_trace.token_usage.append(usage)

        if self.include_costs:
            self._logger.debug(
                f"Token usage: {model} - {input_tokens} in / {output_tokens} out "
                f"(${usage.cost:.4f})"
            )

    def log_api_call(
        self,
        component: str,
        model: str,
        duration_ms: float,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """
        Log an OpenAI API call with timing and token information.

        Args:
            component: Component making the call (e.g., 'routing', 'sql_generation', 'synthesis')
            model: Model name (e.g., 'gpt-4')
            duration_ms: API call duration in milliseconds
            input_tokens: Number of input tokens (if available)
            output_tokens: Number of output tokens (if available)
            success: Whether the call succeeded
            error: Error message if failed
        """
        # Build token info string
        token_info = ""
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
            token_info = f" | tokens: {input_tokens} in + {output_tokens} out = {total_tokens} total"

            # Also log to token usage tracking
            self.log_token_usage(model, input_tokens, output_tokens)

        status = "SUCCESS" if success else "FAILED"
        message = (
            f"[API] {component} | model={model} | {status} | "
            f"duration={duration_ms:.0f}ms{token_info}"
        )
        if error:
            message += f" | error={error}"

        level = "INFO" if success else "WARNING"
        self._log_event(
            "api_call",
            message,
            metadata={
                "component": component,
                "model": model,
                "duration_ms": duration_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "success": success,
                "error": error,
            },
            level=level,
        )

    def log_error(self, error: str, exception: Optional[Exception] = None) -> None:
        """
        Log an error.

        Args:
            error: Error message
            exception: Optional exception object
        """
        self._log_event(
            "error",
            error,
            metadata={"exception": str(exception) if exception else None},
            level="ERROR",
        )

    def log_info(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an info message.

        Args:
            message: Info message
            metadata: Optional metadata dict
        """
        self._log_event(
            "info",
            message,
            metadata=metadata,
            level="INFO",
        )

    def log_semantic_evaluation(
        self,
        verdict: str,
        confidence: float,
        reasoning: str,
        flagged_for_review: bool = False,
        review_reason: Optional[str] = None,
    ) -> None:
        """
        Log semantic evaluation result.

        Args:
            verdict: Evaluation verdict (correct, incorrect, partial, etc.)
            confidence: Confidence score (0.0-1.0)
            reasoning: Reasoning for the verdict
            flagged_for_review: Whether flagged for human review
            review_reason: Reason for flagging
        """
        confidence_pct = confidence * 100
        review_flag = " [FLAGGED FOR REVIEW]" if flagged_for_review else ""
        message = (
            f"Semantic evaluation: {verdict} ({confidence_pct:.0f}% confidence){review_flag}"
        )
        if flagged_for_review and review_reason:
            message += f" | Reason: {review_reason}"

        level = "INFO" if verdict == "correct" else "WARNING"
        self._log_event(
            "semantic_eval",
            message,
            metadata={
                "verdict": verdict,
                "confidence": confidence,
                "reasoning": reasoning,
                "flagged_for_review": flagged_for_review,
                "review_reason": review_reason,
            },
            level=level,
        )

    def _log_event(
        self,
        event_type: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        level: str = "INFO",
    ) -> None:
        """Internal method to log an event."""
        event = TraceEvent(
            event_type=event_type,
            message=message,
            metadata=metadata or {},
            level=level,
        )

        if self.current_trace:
            self.current_trace.events.append(event)

        # Log to handlers
        log_method = getattr(self._logger, level.lower())
        log_method(f"[{event_type}] {message}")

    def _write_trace_json(self) -> None:
        """Write full trace as JSON to separate trace file."""
        if self.current_trace is None or self.log_file is None:
            return

        trace_file = self.log_file.with_suffix(".traces.jsonl")
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.current_trace.to_dict()) + "\n")
