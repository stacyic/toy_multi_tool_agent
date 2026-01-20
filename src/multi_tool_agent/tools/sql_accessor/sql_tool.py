"""Main SQL Accessor tool with retry logic and semantic validation."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...core.exceptions import SQLRetryExhaustedError, SQLValidationError
from .human_review import (
    AmbiguousQueryDetector,
    HumanReviewQueue,
    ReviewReason,
)
from .pii_masker import PIIMasker
from .query_checker import QueryChecker
from .query_executor import QueryExecutor
from .query_generator import QueryGenerator
from .semantic_evaluator import SemanticEvaluation, SemanticEvaluator, SemanticVerdict

if TYPE_CHECKING:
    from ...logging.trace_logger import TraceLogger

# Module-level logger for SQL operations
logger = logging.getLogger("multi_tool_agent.sql")


@dataclass
class SQLAccessorResult:
    """Result from SQLAccessor execution."""

    success: bool
    data: Optional[str] = None
    error: Optional[str] = None
    sql_executed: Optional[str] = None
    attempts: int = 0
    pii_masked: Optional[Dict[str, int]] = None
    # Semantic evaluation fields
    semantic_verdict: Optional[str] = None
    semantic_confidence: Optional[float] = None
    semantic_reasoning: Optional[str] = None
    flagged_for_review: bool = False
    review_reason: Optional[str] = None


class SQLAccessor:
    """
    Text-to-SQL tool with self-correction and semantic validation.

    Features:
    - LLM-based SQL generation with schema awareness
    - AST-based pre-validation (doesn't count toward retries)
    - Self-correction with error feedback (max 3 execution retries)
    - Semantic evaluation using LLM-as-judge
    - Ambiguous query detection and human review queue
    - PII masking on results
    - Business context injection from PolicyAccessor
    """

    def __init__(
        self,
        db_path: str,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        max_generation_retries: int = 3,
        trace_logger: Optional["TraceLogger"] = None,
        # New semantic evaluation options
        enable_semantic_eval: bool = False,
        semantic_eval_model: Optional[str] = None,
        enable_human_review: bool = False,
        review_queue_path: Optional[Path] = None,
        semantic_confidence_threshold: float = 0.7,
    ):
        """
        Initialize the SQL Accessor.

        Args:
            db_path: Path to SQLite database
            model: OpenAI model for SQL generation
            api_key: OpenAI API key
            max_retries: Maximum execution retries (for runtime SQL errors)
            max_generation_retries: Maximum generation/validation retries before giving up
            trace_logger: Optional TraceLogger for structured logging
            enable_semantic_eval: Enable LLM-based semantic evaluation
            semantic_eval_model: Model for semantic evaluation (defaults to same as generator)
            enable_human_review: Enable recording ambiguous queries for human review
            review_queue_path: Path to human review queue file
            semantic_confidence_threshold: Minimum confidence to accept query as correct
        """
        self.max_retries = max_retries
        self.max_generation_retries = max_generation_retries
        self.trace_logger = trace_logger
        self.enable_semantic_eval = enable_semantic_eval
        self.enable_human_review = enable_human_review
        self.semantic_confidence_threshold = semantic_confidence_threshold

        # Initialize core components
        self.executor = QueryExecutor(db_path)
        self.generator = QueryGenerator(model=model, api_key=api_key, logger=trace_logger)
        self.pii_masker = PIIMasker()

        # Initialize checker with schema from database
        schema = self.executor.introspect_schema()
        schema_dict = {
            table: [{"name": col.name, "type": col.type} for col in columns]
            for table, columns in schema.items()
        }
        self.checker = QueryChecker(schema_dict)

        # Set schema for generator
        schema_description = self.executor.get_schema_description()
        self.generator.set_schema(schema_description)

        # Initialize semantic evaluator (optional)
        self.semantic_evaluator: Optional[SemanticEvaluator] = None
        if enable_semantic_eval:
            eval_model = semantic_eval_model or model
            self.semantic_evaluator = SemanticEvaluator(
                model=eval_model,
                api_key=api_key,
                logger=trace_logger,
                confidence_threshold=semantic_confidence_threshold,
            )
            self.semantic_evaluator.set_schema(schema_description)

        # Initialize human review queue (optional)
        self.review_queue: Optional[HumanReviewQueue] = None
        self.ambiguity_detector: Optional[AmbiguousQueryDetector] = None
        if enable_human_review:
            self.review_queue = HumanReviewQueue(
                queue_path=review_queue_path or Path("data/review_queue.json")
            )
            self.ambiguity_detector = AmbiguousQueryDetector()
            # Try to load existing queue
            try:
                self.review_queue.load()
            except Exception as e:
                logger.warning(f"Could not load existing review queue: {e}")

    def _log_sql(
        self,
        attempt: int,
        sql: str,
        success: bool,
        error: Optional[str] = None,
        stage: str = "execution",
    ) -> None:
        """
        Log SQL query attempt to trace logger or module logger.

        Args:
            attempt: Attempt number
            sql: The SQL query
            success: Whether the operation succeeded
            error: Error message if failed
            stage: Stage of processing (generation, validation, execution)
        """
        # Log to trace logger if available (preferred)
        if self.trace_logger:
            if stage == "validation":
                self.trace_logger.log_sql_validation(
                    is_valid=success,
                    errors=[error] if error else [],
                    sql=sql,
                )
            else:
                self.trace_logger.log_sql_attempt(
                    attempt=attempt,
                    sql=sql,
                    success=success,
                    error=error,
                )
        else:
            # Fallback to module logger when trace_logger not available
            sql_oneline = " ".join(sql.split()) if sql else "(empty)"
            if success:
                logger.info(
                    f"[SQL {stage.upper()}] attempt={attempt} | SUCCESS | sql={sql_oneline}"
                )
            else:
                logger.warning(
                    f"[SQL {stage.upper()}] attempt={attempt} | FAILED | error={error} | sql={sql_oneline}"
                )

    async def run(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> SQLAccessorResult:
        """
        Execute text-to-SQL with integrated semantic validation and retry logic.

        Flow: Generate → Validate → Semantic Eval → Execute
        Retries on: syntactic errors, semantic failures, execution errors

        Args:
            question: Natural language question
            context: Optional context from upstream tools (e.g., policy context)
            session_id: Optional session identifier for tracking

        Returns:
            SQLAccessorResult with data or error
        """
        # Pre-check for ambiguous queries
        if self.ambiguity_detector:
            analysis = self.ambiguity_detector.analyze(question)
            if analysis["needs_review"] and self.review_queue:
                should_flag, flag_reason = self.ambiguity_detector.should_flag_for_review(
                    question
                )
                if should_flag:
                    logger.info(f"Question flagged for review: {flag_reason.value}")

        # Extract business context from upstream tools
        business_context = None
        if context and "policy_accessor" in context:
            business_context = context["policy_accessor"]

        attempt_count = 0
        error_feedback = None
        previous_sql = None
        last_error = None
        semantic_evaluation: Optional[SemanticEvaluation] = None

        # Unified retry loop: Generate → Validate → Semantic Eval → Execute
        while attempt_count < self.max_retries:
            attempt_count += 1

            # Step 1: Generate SQL
            try:
                gen_result = await self.generator.generate(
                    question=question,
                    business_context=business_context,
                    error_feedback=error_feedback,
                    previous_sql=previous_sql,
                )
            except Exception as e:
                self._log_sql(
                    attempt=attempt_count,
                    sql=previous_sql or "",
                    success=False,
                    error=f"Generation failed: {str(e)}",
                    stage="generation",
                )
                error_feedback = f"Generation error: {str(e)}"
                continue

            sql = gen_result.sql
            if not sql:
                self._log_sql(
                    attempt=attempt_count,
                    sql="",
                    success=False,
                    error="Generated SQL was empty",
                    stage="generation",
                )
                error_feedback = "Generated SQL was empty. Please generate a valid SELECT query."
                continue

            # Step 2: Syntactic validation
            validation = self.checker.validate(sql)
            if not validation.is_valid:
                error_feedback = self.checker.get_error_summary(validation)
                self._log_sql(
                    attempt=attempt_count,
                    sql=sql,
                    success=False,
                    error=error_feedback,
                    stage="validation",
                )

                # Check for blocked operations - don't retry these
                blocked_ops = ["Update", "Delete", "Drop", "Insert", "Create", "Alter"]
                is_blocked = any(
                    f"{op} operations are not allowed" in error
                    for error in validation.errors
                    for op in blocked_ops
                )

                if is_blocked:
                    op_type = next(
                        (op for op in blocked_ops
                         if any(f"{op} operations are not allowed" in e for e in validation.errors)),
                        "This"
                    )
                    return SQLAccessorResult(
                        success=False,
                        error=f"{op_type} operations are not permitted. This system only supports read-only queries.",
                        sql_executed=sql,
                        attempts=attempt_count,
                    )

                last_error = error_feedback
                previous_sql = sql
                continue

            # Step 3: Semantic evaluation BEFORE execution (if enabled)
            if self.semantic_evaluator:
                try:
                    semantic_evaluation = await self.semantic_evaluator.evaluate(
                        question=question,
                        sql=sql,
                        execution_result=None,  # No execution result yet
                    )

                    # Log semantic evaluation
                    if self.trace_logger:
                        self.trace_logger.log_semantic_evaluation(
                            verdict=semantic_evaluation.verdict.value,
                            confidence=semantic_evaluation.confidence,
                            reasoning=semantic_evaluation.reasoning,
                            flagged_for_review=False,
                            review_reason=None,
                        )

                    # If semantic evaluation fails, get feedback and retry
                    if semantic_evaluation.verdict in [
                        SemanticVerdict.INCORRECT,
                        SemanticVerdict.INVALID_QUESTION,
                    ]:
                        # Get feedback for regeneration
                        feedback = self.semantic_evaluator.get_feedback_for_regeneration(
                            semantic_evaluation
                        )
                        if feedback:
                            error_feedback = feedback
                            previous_sql = sql
                            last_error = f"Semantic evaluation: {semantic_evaluation.verdict.value}"
                            logger.info(
                                f"Semantic evaluation failed ({semantic_evaluation.verdict.value}), "
                                f"retrying with feedback..."
                            )
                            continue

                except Exception as e:
                    logger.warning(f"Semantic evaluation failed: {e}")
                    # Continue to execution even if semantic eval fails

            # Step 4: Execute SQL
            exec_result = await self.executor.execute(sql)

            if not exec_result.success:
                # Execution failed - feed error back for retry
                self._log_sql(
                    attempt=attempt_count,
                    sql=sql,
                    success=False,
                    error=exec_result.error,
                    stage="execution",
                )
                error_feedback = f"SQLite error: {exec_result.error}"
                previous_sql = sql
                last_error = exec_result.error
                continue

            # Execution successful
            self._log_sql(
                attempt=attempt_count,
                sql=sql,
                success=True,
                stage="execution",
            )

            # Step 5: Mask PII in results
            masked = self.pii_masker.mask_sql_result(
                columns=exec_result.columns,
                rows=exec_result.rows,
            )

            if masked.pii_stats and self.trace_logger:
                self.trace_logger.log_pii_masked(masked.pii_stats)

            # Format results
            formatted = self._format_result(
                columns=masked.columns,
                rows=masked.rows,
                row_count=exec_result.row_count,
                requested_columns=gen_result.requested_columns,
            )

            # Step 6: Post-execution semantic evaluation (for final verdict)
            flagged_for_review = False
            review_reason = None

            if self.semantic_evaluator:
                try:
                    # Re-evaluate with execution results for final verdict
                    semantic_evaluation = await self.semantic_evaluator.evaluate(
                        question=question,
                        sql=sql,
                        execution_result=formatted[:500] if formatted else None,
                    )

                    # Check if needs human review (any non-correct result)
                    if self.semantic_evaluator.needs_human_review(semantic_evaluation):
                        flagged_for_review = True
                        # Determine review reason
                        if semantic_evaluation.verdict == SemanticVerdict.AMBIGUOUS:
                            review_reason = ReviewReason.AMBIGUOUS_QUESTION.value
                        elif semantic_evaluation.verdict == SemanticVerdict.INVALID_QUESTION:
                            review_reason = ReviewReason.INVALID_QUESTION.value
                        elif semantic_evaluation.verdict == SemanticVerdict.INCORRECT:
                            review_reason = ReviewReason.SEMANTIC_MISMATCH.value
                        elif semantic_evaluation.verdict == SemanticVerdict.PARTIAL:
                            review_reason = ReviewReason.SEMANTIC_MISMATCH.value
                        elif semantic_evaluation.confidence < self.semantic_confidence_threshold:
                            review_reason = ReviewReason.LOW_CONFIDENCE.value
                        elif semantic_evaluation.question_clarity_score < 0.5:
                            review_reason = ReviewReason.UNCLEAR_INTENT.value
                        else:
                            review_reason = ReviewReason.SEMANTIC_MISMATCH.value

                        # Add to review queue
                        if self.review_queue:
                            self.review_queue.add_for_review(
                                question=question,
                                reason=ReviewReason(review_reason),
                                generated_sql=sql,
                                semantic_evaluation=semantic_evaluation,
                                execution_result=formatted[:200] if formatted else None,
                                session_id=session_id,
                            )
                            # Save queue periodically
                            try:
                                self.review_queue.save()
                            except Exception as e:
                                logger.warning(f"Failed to save review queue: {e}")

                    # Log semantic evaluation (after determining review status)
                    if self.trace_logger:
                        self.trace_logger.log_semantic_evaluation(
                            verdict=semantic_evaluation.verdict.value,
                            confidence=semantic_evaluation.confidence,
                            reasoning=semantic_evaluation.reasoning,
                            flagged_for_review=flagged_for_review,
                            review_reason=review_reason,
                        )

                    # If semantic evaluation found issues but query ran,
                    # include warning in result
                    if semantic_evaluation.verdict not in [
                        SemanticVerdict.CORRECT,
                        SemanticVerdict.PARTIAL,
                    ]:
                        # Add semantic warning to formatted output
                        formatted += (
                            f"\n\n⚠️ Note: The query executed successfully, but "
                            f"there may be semantic issues: {semantic_evaluation.reasoning}"
                        )

                except Exception as e:
                    logger.warning(f"Semantic evaluation failed: {e}")
                    semantic_evaluation = None

            # Return successful result
            return SQLAccessorResult(
                success=True,
                data=formatted,
                sql_executed=sql,
                attempts=attempt_count,
                pii_masked=masked.pii_stats if masked.pii_stats else None,
                semantic_verdict=semantic_evaluation.verdict.value
                if semantic_evaluation else None,
                semantic_confidence=semantic_evaluation.confidence
                if semantic_evaluation else None,
                semantic_reasoning=semantic_evaluation.reasoning
                if semantic_evaluation else None,
                flagged_for_review=flagged_for_review,
                review_reason=review_reason,
            )

        # Exhausted retries - flag for human review
        final_error = error_feedback or last_error or "Unknown error"

        # Add to review queue for repeated failures
        if self.review_queue:
            self.review_queue.add_for_review(
                question=question,
                reason=ReviewReason.REPEATED_FAILURES,
                generated_sql=previous_sql,
                error_message=final_error,
                session_id=session_id,
            )
            try:
                self.review_queue.save()
            except Exception as e:
                logger.warning(f"Failed to save review queue: {e}")

        if self.trace_logger:
            self.trace_logger.log_error(
                f"SQL retries exhausted after {attempt_count} attempts: {final_error}"
            )
        else:
            logger.error(
                f"[SQL EXHAUSTED] attempts={attempt_count} | error={final_error} | "
                f"last_sql={' '.join((previous_sql or '').split())}"
            )

        # Return user-friendly error instead of raising exception
        # Determine if error was generation-related or execution-related
        if "generation" in final_error.lower():
            user_error = (
                f"SQL generation failed after {attempt_count} attempts. "
                f"Last error: {final_error}"
            )
        elif "empty" in final_error.lower():
            user_error = (
                f"I was unable to generate a valid database query after {attempt_count} attempts. "
                f"Please try rephrasing your question or provide more specific details."
            )
        else:
            user_error = (
                f"The query failed after {attempt_count} attempts. "
                f"Last error: {final_error}"
            )

        return SQLAccessorResult(
            success=False,
            error=user_error,
            sql_executed=previous_sql,
            attempts=attempt_count,
            flagged_for_review=True,
            review_reason=ReviewReason.REPEATED_FAILURES.value if self.review_queue else None,
        )

    def _format_result(
        self,
        columns: List[str],
        rows: List[tuple],
        row_count: int,
        requested_columns: List[str],
    ) -> str:
        """
        Format query results as a readable aligned table.

        Args:
            columns: Column names
            rows: Result rows
            row_count: Total row count
            requested_columns: Columns that were requested (for validation)

        Returns:
            Formatted result string
        """
        if not rows:
            return "No results found."

        # Limit display rows
        display_rows = rows[:50]

        # Convert all values to strings and handle None
        str_rows = []
        for row in display_rows:
            str_row = []
            for v in row:
                if v is None:
                    str_row.append("NULL")
                else:
                    # Truncate long values for display
                    s = str(v)
                    if len(s) > 50:
                        s = s[:47] + "..."
                    str_row.append(s)
            str_rows.append(str_row)

        # Calculate column widths (min 3, max 50)
        col_widths = []
        for i, col in enumerate(columns):
            max_width = len(col)
            for row in str_rows:
                if i < len(row):
                    max_width = max(max_width, len(row[i]))
            col_widths.append(min(max(max_width, 3), 50))

        # Build the table
        lines = [f"Results ({row_count} rows):", ""]

        # Header row
        header_parts = []
        for i, col in enumerate(columns):
            header_parts.append(col.ljust(col_widths[i]))
        lines.append(" | ".join(header_parts))

        # Separator row
        sep_parts = ["-" * w for w in col_widths]
        lines.append("-+-".join(sep_parts))

        # Data rows
        for str_row in str_rows:
            row_parts = []
            for i, val in enumerate(str_row):
                if i < len(col_widths):
                    row_parts.append(val.ljust(col_widths[i]))
                else:
                    row_parts.append(val)
            lines.append(" | ".join(row_parts))

        if row_count > 50:
            lines.append("")
            lines.append(f"... and {row_count - 50} more rows")

        return "\n".join(lines)

    def get_schema_description(self) -> str:
        """Get the database schema description."""
        return self.executor.get_schema_description()

    def get_review_queue_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get statistics about the human review queue.

        Returns:
            Queue statistics dict, or None if human review is disabled
        """
        if not self.review_queue:
            return None
        return self.review_queue.get_statistics()

    def export_review_queue(self, output_path: Path) -> None:
        """
        Export pending review items to a markdown file for human review.

        Args:
            output_path: Path to output markdown file
        """
        if not self.review_queue:
            raise ValueError("Human review is not enabled")
        self.review_queue.export_for_review(output_path)

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """
        Get list of pending review items.

        Returns:
            List of pending review items as dicts
        """
        if not self.review_queue:
            return []
        from dataclasses import asdict
        return [
            asdict(item)
            for item in self.review_queue.get_pending_items()
        ]
