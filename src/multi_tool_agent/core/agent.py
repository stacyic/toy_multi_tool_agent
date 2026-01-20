"""Agent orchestration with dependency-aware tool execution."""

import asyncio
import re
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from ..logging.performance_logger import Component, PerformanceLogger
from ..logging.trace_logger import TraceLogger
from ..tools.policy_accessor import PolicyAccessor
from ..tools.rag_metrics import RAGMetricsTracker
from ..tools.sql_accessor.sql_tool import SQLAccessor
from ..utils.llm_utils import invoke_with_logging
from .batch_processor import (
    BatchMetrics,
    BatchProcessor,
    BatchQueryResult,
    BatchResult,
    QueryMetrics,
    Timer,
)
from .exceptions import AgentError, RoutingError, ToolExecutionError
from .query_router import QueryRouter, RoutingDecision


class MultiToolAgent:
    """
    Multi-tool agent with dependency-aware orchestration.

    Features:
    - Query routing to determine needed tools
    - Dependency-based execution order
    - Context passing between tools
    - Graceful degradation on failures
    - Cost tracking and logging
    """

    def __init__(
        self,
        db_path: Path,
        policies_path: Path,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        logger: Optional[TraceLogger] = None,
        perf_logger: Optional[PerformanceLogger] = None,
        graceful_degradation: bool = True,
        max_sql_retries: int = 3,
        max_generation_retries: int = 3,
        # SQL semantic evaluation options
        enable_semantic_eval: bool = False,
        semantic_eval_model: Optional[str] = None,
        enable_human_review: bool = False,
        review_queue_path: Optional[Path] = None,
        semantic_confidence_threshold: float = 0.7,
        # RAG metrics options
        enable_rag_metrics: bool = False,
        rag_metrics_path: Optional[Path] = None,
    ):
        """
        Initialize the multi-tool agent.

        Args:
            db_path: Path to SQLite database
            policies_path: Path to policies markdown file
            model: OpenAI model for SQL generation and routing
            api_key: OpenAI API key
            logger: Optional TraceLogger for logging
            perf_logger: Optional PerformanceLogger for detailed timing
            graceful_degradation: Enable graceful degradation on failures
            max_sql_retries: Maximum SQL execution retries for runtime errors
            max_generation_retries: Maximum SQL generation retries before giving up
            enable_semantic_eval: Enable LLM-based semantic evaluation of SQL queries
            semantic_eval_model: Model for semantic evaluation (defaults to same as generator)
            enable_human_review: Enable recording ambiguous queries for human review
            review_queue_path: Path to human review queue file
            semantic_confidence_threshold: Minimum confidence to accept query as correct
            enable_rag_metrics: Enable RAG retrieval metrics tracking
            rag_metrics_path: Path to export RAG metrics JSON
        """
        self.model = model
        self.api_key = api_key
        self.graceful_degradation = graceful_degradation
        self.logger = logger
        self.perf_logger = perf_logger
        self.enable_semantic_eval = enable_semantic_eval
        self.enable_rag_metrics = enable_rag_metrics

        # Initialize RAG metrics tracker if enabled
        self.rag_metrics_tracker: Optional[RAGMetricsTracker] = None
        if enable_rag_metrics:
            self.rag_metrics_tracker = RAGMetricsTracker(metrics_path=rag_metrics_path)

        # Initialize components
        self.router = QueryRouter(model=model, api_key=api_key, logger=logger)

        self.policy_accessor = PolicyAccessor(
            policies_path=policies_path,
            api_key=api_key,
            metrics_tracker=self.rag_metrics_tracker,
        )

        self.sql_accessor = SQLAccessor(
            db_path=str(db_path),
            model=model,
            api_key=api_key,
            max_retries=max_sql_retries,
            max_generation_retries=max_generation_retries,
            trace_logger=logger,
            enable_semantic_eval=enable_semantic_eval,
            semantic_eval_model=semantic_eval_model,
            enable_human_review=enable_human_review,
            review_queue_path=review_queue_path,
            semantic_confidence_threshold=semantic_confidence_threshold,
        )

        # LLM for response synthesis
        kwargs = {"model": model, "temperature": 0.3}
        if api_key:
            kwargs["api_key"] = api_key
        self._synthesis_llm = ChatOpenAI(**kwargs)

        # Tool registry
        self._tools: Dict[str, Any] = {
            "policy_accessor": self.policy_accessor,
            "sql_accessor": self.sql_accessor,
        }

        # Batch processor for optimized batch queries
        self._batch_processor = BatchProcessor(
            router=self.router,
            max_concurrent=5,
        )

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize agent components (load policies, etc.)."""
        if self._initialized:
            return

        await self.policy_accessor.initialize()
        self._initialized = True

    def _detect_sql_injection(self, query: str) -> Optional[str]:
        """
        Detect SQL injection attempts in user's natural language query.

        Args:
            query: User's natural language query

        Returns:
            Rejection message if injection detected, None otherwise
        """
        # SQL keywords that indicate malicious intent when in natural language
        dangerous_patterns = [
            # Direct SQL commands
            (r'\bDROP\s+TABLE\b', 'DROP TABLE'),
            (r'\bDROP\s+DATABASE\b', 'DROP DATABASE'),
            (r'\bDELETE\s+FROM\b', 'DELETE FROM'),
            (r'\bTRUNCATE\s+TABLE\b', 'TRUNCATE TABLE'),
            (r'\bINSERT\s+INTO\b', 'INSERT INTO'),
            (r'\bUPDATE\s+\w+\s+SET\b', 'UPDATE...SET'),
            (r'\bALTER\s+TABLE\b', 'ALTER TABLE'),
            (r'\bCREATE\s+TABLE\b', 'CREATE TABLE'),
            (r'\bEXEC\s*\(', 'EXEC'),
            (r'\bEXECUTE\s*\(', 'EXECUTE'),
            # Chained SQL commands (injection via semicolon)
            (r';\s*DROP\b', 'chained DROP command'),
            (r';\s*DELETE\b', 'chained DELETE command'),
            (r';\s*UPDATE\b', 'chained UPDATE command'),
            (r';\s*INSERT\b', 'chained INSERT command'),
            (r';\s*ALTER\b', 'chained ALTER command'),
            (r';\s*CREATE\b', 'chained CREATE command'),
            (r';\s*TRUNCATE\b', 'chained TRUNCATE command'),
            # Common injection patterns
            (r'--\s*$', 'SQL comment injection'),
            (r"'\s*OR\s+'1'\s*=\s*'1", 'OR injection'),
            (r"'\s*OR\s+1\s*=\s*1", 'OR injection'),
            (r';\s*--', 'comment-based injection'),
            # Natural language requests for destructive operations
            (r'\b(delete|remove|erase|destroy)\s+(all|every|the)\s+\w*\s*(record|row|data|entr)', 'destructive operation request'),
            (r'\b(drop|delete|remove)\s+(the\s+)?(table|database)\b', 'destructive operation request'),
            (r'\bwipe\s+(the\s+)?(table|database|data)\b', 'destructive operation request'),
        ]

        detected_threats = []
        for pattern, threat_name in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                detected_threats.append(threat_name)

        if detected_threats:
            threats_str = ", ".join(detected_threats)
            return (
                f"**Security Alert: Request Rejected**\n\n"
                f"Your query contains potentially malicious SQL patterns that have been detected and blocked:\n"
                f"- Detected: {threats_str}\n\n"
                f"This system only supports read-only queries for retrieving information. "
                f"Operations that modify or delete data are not permitted.\n\n"
                f"If you have a legitimate question, please rephrase it without SQL commands."
            )

        return None

    async def execute(self, query: str) -> str:
        """
        Execute a user query.

        Args:
            query: Natural language query

        Returns:
            Response string
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        # Generate request ID
        request_id = str(uuid.uuid4())[:8]

        # Start trace and performance logging
        if self.logger:
            self.logger.start_request(request_id, query)
        if self.perf_logger:
            self.perf_logger.start_request(request_id, query)

        # Check for SQL injection attempts in the user query
        injection_check = self._detect_sql_injection(query)
        if injection_check:
            if self.logger:
                self.logger.log_error(f"SQL injection attempt detected: {injection_check}")
                self.logger.end_request(success=False, error="SQL injection detected")
            if self.perf_logger:
                self.perf_logger.end_request(success=False, error="SQL injection detected")
            return injection_check

        try:
            # Step 1: Route query
            if self.perf_logger:
                with self.perf_logger.time_component(Component.ROUTING):
                    decision = await self._route_query(query)
                self.perf_logger.log_tool_selection(decision.tools, decision.reasoning)
            else:
                decision = await self._route_query(query)

            if self.logger:
                self.logger.log_tool_selection(
                    decision.tools,
                    decision.reasoning,
                )

            # Step 2: Execute tools in dependency order
            context = await self._execute_tools(query, decision)

            # Step 3: Synthesize response
            if self.perf_logger:
                with self.perf_logger.time_component(Component.RESPONSE_SYNTHESIS):
                    response = await self._synthesize_response(query, context, decision)
            else:
                response = await self._synthesize_response(query, context, decision)

            # End trace
            if self.logger:
                self.logger.end_request(success=True)
            if self.perf_logger:
                self.perf_logger.end_request(success=True)

            return response

        except Exception as e:
            if self.logger:
                self.logger.log_error(str(e), e)
                self.logger.end_request(success=False, error=str(e))
            if self.perf_logger:
                self.perf_logger.end_request(success=False, error=str(e))

            if self.graceful_degradation:
                return self._generate_fallback_response(query, e)
            raise

    async def _route_query(self, query: str) -> RoutingDecision:
        """Route query to appropriate tools."""
        try:
            return await self.router.route(query)
        except Exception as e:
            if self.graceful_degradation:
                # Default to trying both tools
                return RoutingDecision(
                    tools=["policy_accessor", "sql_accessor"],
                    reasoning="Fallback routing due to error",
                    requires_context_passing=True,
                )
            raise RoutingError(f"Query routing failed: {e}")

    async def _execute_tools(
        self,
        query: str,
        decision: RoutingDecision,
    ) -> Dict[str, Any]:
        """
        Execute tools in dependency order.

        Args:
            query: User query
            decision: Routing decision

        Returns:
            Context dictionary with tool results
        """
        context: Dict[str, Any] = {}

        # Resolve execution order
        ordered_tools = QueryRouter.resolve_execution_order(decision.tools)

        for tool_name in ordered_tools:
            try:
                result = await self._execute_single_tool(
                    tool_name,
                    query,
                    context,
                    decision.requires_context_passing,
                )
                context[tool_name] = result

            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"Tool {tool_name} failed: {e}", e)

                if self.graceful_degradation:
                    context[tool_name] = {
                        "error": str(e),
                        "success": False,
                    }
                else:
                    raise ToolExecutionError(str(e), tool_name=tool_name)

        return context

    async def _execute_single_tool(
        self,
        tool_name: str,
        query: str,
        context: Dict[str, Any],
        pass_context: bool,
    ) -> Dict[str, Any]:
        """
        Execute a single tool.

        Args:
            tool_name: Name of tool to execute
            query: User query
            context: Current context from previous tools
            pass_context: Whether to pass context to this tool

        Returns:
            Tool result dictionary
        """
        if tool_name == "policy_accessor":
            if self.perf_logger:
                with self.perf_logger.time_component(Component.POLICY_SEARCH, query=query):
                    result = await self.policy_accessor.search(query)
            else:
                result = await self.policy_accessor.search(query)

            if self.logger:
                self.logger.log_policy_search(
                    query,
                    len(result.sources),
                    result.from_cache,
                )

            return {
                "content": result.content,
                "sources": result.sources,
                "from_cache": result.from_cache,
                "success": True,
            }

        elif tool_name == "sql_accessor":
            # Build context for SQL generation
            sql_context = None
            if pass_context and "policy_accessor" in context:
                policy_result = context["policy_accessor"]
                if policy_result.get("success"):
                    sql_context = {
                        "policy_accessor": policy_result.get("content", ""),
                    }

            # Execute SQL with performance timing
            start_time = time.perf_counter()
            if self.perf_logger:
                with self.perf_logger.time_component(
                    Component.SQL_EXECUTION, query=query
                ):
                    result = await self.sql_accessor.run(query, context=sql_context)
            else:
                result = await self.sql_accessor.run(query, context=sql_context)
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Log to perf_logger only (SQLAccessor handles trace_logger internally)
            if self.perf_logger:
                self.perf_logger.log_sql_attempt(
                    attempt=result.attempts,
                    sql=result.sql_executed or "",
                    success=result.success,
                    error=result.error,
                    execution_time_ms=execution_time_ms,
                )
                if result.pii_masked:
                    self.perf_logger.log_pii_masked(result.pii_masked)

            return {
                "data": result.data,
                "sql": result.sql_executed,
                "attempts": result.attempts,
                "pii_masked": result.pii_masked,
                "success": result.success,
                "error": result.error,
            }

        else:
            raise ToolExecutionError(f"Unknown tool: {tool_name}", tool_name=tool_name)

    async def _synthesize_response(
        self,
        query: str,
        context: Dict[str, Any],
        decision: RoutingDecision,
    ) -> str:
        """
        Synthesize final response from tool results.

        Args:
            query: Original query
            context: Results from all executed tools
            decision: Routing decision

        Returns:
            Synthesized response string
        """
        # Check if we need to ask for clarification (multiple matching people)
        clarification = self._check_multiple_person_matches(query, context)
        if clarification:
            return clarification

        # Handle single-tool responses
        if len(decision.tools) == 1:
            tool_name = decision.tools[0]
            result = context.get(tool_name, {})

            if not result.get("success"):
                return f"I encountered an error: {result.get('error', 'Unknown error')}"

            # All responses go through LLM synthesis for natural language phrasing

        # Use LLM to synthesize natural language response from tool results
        return await self._llm_synthesize(query, context)

    def _check_multiple_person_matches(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """
        Check if SQL results contain multiple people matching a name query.

        If the user asked about a specific person (e.g., "Alex's email") but
        multiple people match, return a clarification message.

        Args:
            query: Original user query
            context: Tool execution results

        Returns:
            Clarification message if multiple matches, None otherwise
        """
        # Check if this is a person-specific query
        # Look for patterns like "Alex's", "customer Alex", "Alex Garcia", etc.
        person_patterns = [
            r"\b([A-Z][a-z]+)'s\b",  # "Alex's email"
            r"\bcustomer\s+([A-Z][a-z]+)\b",  # "customer Alex"
            r"\buser\s+([A-Z][a-z]+)\b",  # "user Alex"
            r"\b([A-Z][a-z]+)\s+(?:email|phone|address|order|info|information)\b",  # "Alex email"
        ]

        query_name = None
        for pattern in person_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                query_name = match.group(1)
                break

        if not query_name:
            return None

        # Check SQL results
        sql_result = context.get("sql_accessor", {})
        if not sql_result.get("success"):
            return None

        data = sql_result.get("data", "")
        if not data or "No results found" in data:
            return None

        # Parse the results to count rows and extract names
        lines = data.strip().split("\n")

        # Find the header line (contains column names)
        header_idx = -1
        for i, line in enumerate(lines):
            if "Results (" in line:
                continue
            if line.strip() and not line.startswith("-"):
                header_idx = i
                break

        if header_idx == -1:
            return None

        # Check if 'name' column exists
        header_line = lines[header_idx] if header_idx < len(lines) else ""
        columns = [col.strip().lower() for col in header_line.split("|")]

        name_col_idx = -1
        for i, col in enumerate(columns):
            if col in ("name", "customer_name", "full_name"):
                name_col_idx = i
                break

        if name_col_idx == -1:
            return None

        # Extract data rows (skip header and separator)
        data_start = header_idx + 2  # Skip header and separator line
        data_rows = []
        for line in lines[data_start:]:
            line = line.strip()
            if not line or line.startswith("..."):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) > name_col_idx:
                data_rows.append(parts)

        # If multiple rows with different names, ask for clarification
        if len(data_rows) > 1:
            names = []
            for row in data_rows:
                if name_col_idx < len(row):
                    name = row[name_col_idx]
                    if name and name not in names:
                        names.append(name)

            if len(names) > 1:
                # Multiple different people found - ask for clarification
                name_list = "\n".join(f"  - {name}" for name in names[:10])
                if len(names) > 10:
                    name_list += f"\n  - ... and {len(names) - 10} more"

                return (
                    f"I found multiple customers matching '{query_name}':\n\n"
                    f"{name_list}\n\n"
                    f"Could you please specify which customer you're asking about? "
                    f"You can use their full name (e.g., 'Alex Garcia') to be more specific."
                )

        return None

    async def _llm_synthesize(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> str:
        """Use LLM to synthesize response from multiple tool results."""
        # Get today's date for time-sensitive calculations
        today = date.today()
        today_str = today.strftime("%B %d, %Y")  # e.g., "December 16, 2025"

        # Build context summary
        parts = [f"User Query: {query}", "", "Tool Results:"]

        if "policy_accessor" in context:
            policy = context["policy_accessor"]
            if policy.get("success"):
                parts.append(f"\n--- Policy Information ---")
                parts.append(policy.get("content", "")[:2000])  # Limit size

        if "sql_accessor" in context:
            sql = context["sql_accessor"]
            if sql.get("success"):
                parts.append(f"\n--- Database Query Results ---")
                parts.append(sql.get("data", "")[:2000])
            else:
                parts.append(f"\n--- Database Query Failed ---")
                parts.append(f"Error: {sql.get('error', 'Unknown error')}")

        context_text = "\n".join(parts)

        # Generate synthesis
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a helpful assistant. Today's date is {today_str}. "
                    "Synthesize the tool results below into a clear, natural language answer to the user's question. "
                    "\n\nGuidelines:"
                    "\n- Convert raw data tables into conversational responses"
                    "\n- For customer/product lookups, summarize the key information naturally"
                    "\n- For counts or aggregates, state the number clearly in a sentence"
                    "\n- If data shows [REDACTED], mention that sensitive information has been protected"
                    "\n- If both policy and data are provided, combine them appropriately"
                    "\n- When calculating time periods (e.g., return windows), use today's date"
                    "\n- Be concise but complete - include all relevant information from the results"
                    "\n- Do not show raw table formatting in your response"
                ),
            },
            {"role": "user", "content": context_text},
        ]

        response = await invoke_with_logging(
            llm=self._synthesis_llm,
            messages=messages,
            logger=self.logger,
            component="synthesis",
            model=self.model,
        )
        return response.content

    def _generate_fallback_response(self, query: str, error: Exception) -> str:
        """Generate a fallback response when errors occur."""
        return (
            f"I apologize, but I encountered an issue processing your request: {error}\n\n"
            "Please try rephrasing your question or contact support if the issue persists."
        )

    def get_schema_description(self) -> str:
        """Get the database schema description."""
        return self.sql_accessor.get_schema_description()

    async def execute_batch(
        self,
        queries: List[str],
        max_concurrent: int = 5,
    ) -> BatchResult:
        """
        Execute multiple queries with optimized batching.

        Optimizations:
        - Shared policy context across queries needing policy info
        - Parallel execution of independent queries
        - Aggregate metrics for batch analysis

        Args:
            queries: List of natural language queries
            max_concurrent: Maximum concurrent query executions

        Returns:
            BatchResult with individual results and aggregate metrics
        """
        if not queries:
            return BatchResult(results=[], batch_metrics=BatchMetrics())

        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        batch_start = time.perf_counter()

        # Step 1: Route all queries
        decisions, routing_time = await self._batch_processor.route_batch(queries)

        # Step 2: Categorize queries by tool requirements
        categories = self._batch_processor.categorize_queries(queries, decisions)

        # Step 3: Pre-fetch shared policy context for queries that need it
        shared_policy_context = None
        policy_fetch_time = 0.0
        policy_needing_count = len(categories["policy_only"]) + len(categories["both"])

        if policy_needing_count > 0:
            # Use a representative query for policy fetch
            representative_query = (
                categories["policy_only"][0][1]
                if categories["policy_only"]
                else categories["both"][0][1]
            )
            with Timer() as policy_timer:
                try:
                    result = await self.policy_accessor.search(representative_query)
                    shared_policy_context = {
                        "content": result.content,
                        "sources": result.sources,
                        "from_cache": result.from_cache,
                        "success": True,
                    }
                except Exception as e:
                    if self.logger:
                        self.logger.log_error(f"Batch policy fetch failed: {e}", e)
                    shared_policy_context = {"success": False, "error": str(e)}
            policy_fetch_time = policy_timer.elapsed

        # Step 4: Execute all queries with shared context
        semaphore = asyncio.Semaphore(max_concurrent)
        results: List[BatchQueryResult] = [None] * len(queries)  # type: ignore

        async def execute_single(
            index: int,
            query: str,
            decision: RoutingDecision,
        ) -> None:
            async with semaphore:
                result = await self._execute_batch_query(
                    index=index,
                    query=query,
                    decision=decision,
                    shared_policy_context=shared_policy_context,
                    routing_time_per_query=routing_time / len(queries),
                )
                results[index] = result

        # Execute all queries concurrently (respecting semaphore limit)
        all_queries_with_decisions = [
            (i, query, decisions[i]) for i, query in enumerate(queries)
        ]

        await asyncio.gather(
            *[
                execute_single(i, q, d)
                for i, q, d in all_queries_with_decisions
            ],
            return_exceptions=True,
        )

        # Handle any None results (from exceptions)
        for i, result in enumerate(results):
            if result is None:
                results[i] = BatchQueryResult(
                    query=queries[i],
                    response="An unexpected error occurred.",
                    success=False,
                    error="Query execution failed",
                    request_id=str(uuid.uuid4())[:8],
                )

        # Step 5: Calculate batch metrics
        batch_time = (time.perf_counter() - batch_start) * 1000
        policy_saved, routing_saved = self._batch_processor.calculate_savings(
            queries, categories
        )

        batch_metrics = BatchMetrics(
            total_queries=len(queries),
            successful_queries=sum(1 for r in results if r.success),
            failed_queries=sum(1 for r in results if not r.success),
            total_time_ms=batch_time,
            avg_time_per_query_ms=batch_time / len(queries) if queries else 0,
            policy_calls_saved=policy_saved,
            routing_calls_saved=routing_saved,
        )

        # Log batch completion
        if self.logger:
            self.logger._log_event(
                "batch_complete",
                f"Batch completed: {batch_metrics.successful_queries}/{batch_metrics.total_queries} "
                f"successful in {batch_metrics.total_time_ms:.0f}ms "
                f"(saved {policy_saved} policy calls)",
                metadata=batch_metrics.to_dict(),
            )

        return BatchResult(results=results, batch_metrics=batch_metrics)

    async def _execute_batch_query(
        self,
        index: int,
        query: str,
        decision: RoutingDecision,
        shared_policy_context: Optional[Dict[str, Any]],
        routing_time_per_query: float,
    ) -> BatchQueryResult:
        """
        Execute a single query within a batch context.

        Args:
            index: Query index in batch
            query: The query to execute
            decision: Routing decision for this query
            shared_policy_context: Pre-fetched policy context (if available)
            routing_time_per_query: Amortized routing time

        Returns:
            BatchQueryResult for this query
        """
        request_id = str(uuid.uuid4())[:8]
        metrics = QueryMetrics(routing_time_ms=routing_time_per_query)
        query_start = time.perf_counter()

        try:
            # Execute tools in dependency order
            context: Dict[str, Any] = {}
            ordered_tools = QueryRouter.resolve_execution_order(decision.tools)

            for tool_name in ordered_tools:
                if tool_name == "policy_accessor":
                    # Use shared context if available
                    if shared_policy_context and shared_policy_context.get("success"):
                        context["policy_accessor"] = shared_policy_context
                        # No additional time since we used shared context
                    else:
                        # Fall back to individual fetch
                        with Timer() as t:
                            result = await self.policy_accessor.search(query)
                            context["policy_accessor"] = {
                                "content": result.content,
                                "sources": result.sources,
                                "from_cache": result.from_cache,
                                "success": True,
                            }
                        metrics.policy_time_ms = t.elapsed

                elif tool_name == "sql_accessor":
                    sql_context = None
                    if decision.requires_context_passing and "policy_accessor" in context:
                        policy_result = context["policy_accessor"]
                        if policy_result.get("success"):
                            sql_context = {"policy_accessor": policy_result.get("content", "")}

                    with Timer() as t:
                        result = await self.sql_accessor.run(query, context=sql_context)
                    metrics.sql_time_ms = t.elapsed
                    metrics.sql_attempts = result.attempts
                    metrics.pii_masked = result.pii_masked

                    context["sql_accessor"] = {
                        "data": result.data,
                        "sql": result.sql_executed,
                        "attempts": result.attempts,
                        "pii_masked": result.pii_masked,
                        "success": result.success,
                        "error": result.error,
                    }

            # Synthesize response
            with Timer() as t:
                response = await self._synthesize_response(query, context, decision)
            metrics.synthesis_time_ms = t.elapsed

            metrics.total_time_ms = (time.perf_counter() - query_start) * 1000

            return BatchQueryResult(
                query=query,
                response=response,
                success=True,
                metrics=metrics,
                request_id=request_id,
            )

        except Exception as e:
            metrics.total_time_ms = (time.perf_counter() - query_start) * 1000

            if self.graceful_degradation:
                return BatchQueryResult(
                    query=query,
                    response=self._generate_fallback_response(query, e),
                    success=False,
                    error=str(e),
                    metrics=metrics,
                    request_id=request_id,
                )
            raise

    def get_rag_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get RAG retrieval metrics.

        Returns:
            Metrics dictionary, or None if RAG metrics not enabled
        """
        if not self.rag_metrics_tracker:
            return None
        return self.rag_metrics_tracker.get_metrics().to_dict()

    def get_rag_metrics_summary(self) -> Optional[str]:
        """
        Get a formatted summary of RAG retrieval metrics.

        Returns:
            Formatted metrics summary, or None if RAG metrics not enabled
        """
        if not self.rag_metrics_tracker:
            return None
        return self.rag_metrics_tracker.get_summary()

    def export_rag_metrics(self, path: Optional[Path] = None) -> None:
        """
        Export RAG metrics to a JSON file.

        Args:
            path: Output path (uses configured path if not provided)
        """
        if self.rag_metrics_tracker:
            self.rag_metrics_tracker.export_metrics(path)
