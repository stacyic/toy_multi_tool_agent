"""Golden set evaluation framework for SQL query accuracy.

This module provides infrastructure for maintaining a golden set of
question-SQL pairs and evaluating SQL generation accuracy against it.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .semantic_evaluator import SemanticEvaluation, SemanticVerdict


logger = logging.getLogger("multi_tool_agent.golden_set")


@dataclass
class GoldenSetEntry:
    """A single entry in the golden set."""

    id: str
    question: str
    expected_sql: str
    expected_tables: List[str]  # Tables that should be queried
    expected_columns: List[str]  # Key columns that should appear
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"  # general, aggregation, join, filter, etc.
    notes: str = ""  # Additional notes about this test case
    is_valid_question: bool = True  # False for intentionally invalid questions
    expected_verdict: str = "correct"  # Expected semantic verdict


@dataclass
class EvaluationResult:
    """Result from evaluating a single golden set entry."""

    entry_id: str
    question: str
    expected_sql: str
    generated_sql: str
    semantic_verdict: SemanticVerdict
    semantic_confidence: float
    syntactic_valid: bool
    tables_match: bool
    columns_match: bool
    execution_success: bool
    execution_match: bool  # Results match expected (if available)
    overall_pass: bool
    error_message: Optional[str] = None
    evaluation_time_ms: float = 0.0


@dataclass
class AccuracyMetrics:
    """Aggregated accuracy metrics from golden set evaluation."""

    total_entries: int
    passed: int
    failed: int

    # Breakdown by verdict
    correct_count: int = 0
    incorrect_count: int = 0
    partial_count: int = 0
    ambiguous_count: int = 0
    invalid_question_count: int = 0

    # Component-level metrics
    syntactic_accuracy: float = 0.0  # % with valid SQL syntax
    semantic_accuracy: float = 0.0  # % semantically correct
    execution_accuracy: float = 0.0  # % successfully executed
    table_accuracy: float = 0.0  # % with correct tables
    column_accuracy: float = 0.0  # % with correct columns

    # Breakdown by category
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Timing
    avg_evaluation_time_ms: float = 0.0
    total_evaluation_time_ms: float = 0.0

    # Metadata
    timestamp: str = ""
    model_used: str = ""

    @property
    def overall_accuracy(self) -> float:
        """Overall pass rate as percentage."""
        if self.total_entries == 0:
            return 0.0
        return (self.passed / self.total_entries) * 100

    @property
    def pass_rate(self) -> float:
        """Alias for overall_accuracy."""
        return self.overall_accuracy


class GoldenSetManager:
    """
    Manages golden set data and evaluation.

    Features:
    - Load/save golden set from JSON files
    - Add/remove entries programmatically
    - Run evaluation against the golden set
    - Track metrics over time
    """

    def __init__(self, golden_set_path: Optional[Path] = None):
        """
        Initialize the golden set manager.

        Args:
            golden_set_path: Path to golden set JSON file (optional)
        """
        self.golden_set_path = golden_set_path or Path("data/golden_set.json")
        self.entries: List[GoldenSetEntry] = []
        self._metrics_history: List[AccuracyMetrics] = []

    def load(self, path: Optional[Path] = None) -> None:
        """Load golden set from JSON file."""
        load_path = path or self.golden_set_path

        if not load_path.exists():
            logger.info(f"Golden set file not found at {load_path}, starting empty")
            return

        try:
            with open(load_path, "r") as f:
                data = json.load(f)

            self.entries = [
                GoldenSetEntry(**entry) for entry in data.get("entries", [])
            ]
            logger.info(f"Loaded {len(self.entries)} golden set entries")

        except Exception as e:
            logger.error(f"Failed to load golden set: {e}")
            raise

    def save(self, path: Optional[Path] = None) -> None:
        """Save golden set to JSON file."""
        save_path = path or self.golden_set_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "entries": [asdict(entry) for entry in self.entries],
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.entries)} golden set entries to {save_path}")

    def add_entry(self, entry: GoldenSetEntry) -> None:
        """Add a new entry to the golden set."""
        # Check for duplicate IDs
        if any(e.id == entry.id for e in self.entries):
            raise ValueError(f"Entry with ID '{entry.id}' already exists")

        self.entries.append(entry)
        logger.info(f"Added golden set entry: {entry.id}")

    def remove_entry(self, entry_id: str) -> bool:
        """Remove an entry by ID."""
        for i, entry in enumerate(self.entries):
            if entry.id == entry_id:
                del self.entries[i]
                logger.info(f"Removed golden set entry: {entry_id}")
                return True
        return False

    def get_entry(self, entry_id: str) -> Optional[GoldenSetEntry]:
        """Get an entry by ID."""
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def get_entries_by_category(self, category: str) -> List[GoldenSetEntry]:
        """Get all entries in a category."""
        return [e for e in self.entries if e.category == category]

    def get_entries_by_difficulty(self, difficulty: str) -> List[GoldenSetEntry]:
        """Get all entries of a difficulty level."""
        return [e for e in self.entries if e.difficulty == difficulty]

    def calculate_metrics(
        self,
        results: List[EvaluationResult],
        model_used: str = "",
    ) -> AccuracyMetrics:
        """
        Calculate accuracy metrics from evaluation results.

        Args:
            results: List of EvaluationResult from evaluation run
            model_used: Model identifier for tracking

        Returns:
            AccuracyMetrics with aggregated statistics
        """
        if not results:
            return AccuracyMetrics(
                total_entries=0,
                passed=0,
                failed=0,
                timestamp=datetime.now().isoformat(),
                model_used=model_used,
            )

        total = len(results)
        passed = sum(1 for r in results if r.overall_pass)

        # Verdict breakdown
        verdict_counts = {v: 0 for v in SemanticVerdict}
        for r in results:
            verdict_counts[r.semantic_verdict] += 1

        # Component accuracies
        syntactic_correct = sum(1 for r in results if r.syntactic_valid)
        semantic_correct = sum(
            1 for r in results
            if r.semantic_verdict == SemanticVerdict.CORRECT
        )
        execution_success = sum(1 for r in results if r.execution_success)
        tables_correct = sum(1 for r in results if r.tables_match)
        columns_correct = sum(1 for r in results if r.columns_match)

        # Category breakdown
        by_category: Dict[str, Dict[str, float]] = {}
        categories = set(self._get_entry_category(r.entry_id) for r in results)
        for cat in categories:
            cat_results = [
                r for r in results
                if self._get_entry_category(r.entry_id) == cat
            ]
            if cat_results:
                cat_passed = sum(1 for r in cat_results if r.overall_pass)
                by_category[cat] = {
                    "total": len(cat_results),
                    "passed": cat_passed,
                    "accuracy": (cat_passed / len(cat_results)) * 100,
                }

        # Difficulty breakdown
        by_difficulty: Dict[str, Dict[str, float]] = {}
        difficulties = set(self._get_entry_difficulty(r.entry_id) for r in results)
        for diff in difficulties:
            diff_results = [
                r for r in results
                if self._get_entry_difficulty(r.entry_id) == diff
            ]
            if diff_results:
                diff_passed = sum(1 for r in diff_results if r.overall_pass)
                by_difficulty[diff] = {
                    "total": len(diff_results),
                    "passed": diff_passed,
                    "accuracy": (diff_passed / len(diff_results)) * 100,
                }

        # Timing
        total_time = sum(r.evaluation_time_ms for r in results)
        avg_time = total_time / total if total > 0 else 0

        metrics = AccuracyMetrics(
            total_entries=total,
            passed=passed,
            failed=total - passed,
            correct_count=verdict_counts[SemanticVerdict.CORRECT],
            incorrect_count=verdict_counts[SemanticVerdict.INCORRECT],
            partial_count=verdict_counts[SemanticVerdict.PARTIAL],
            ambiguous_count=verdict_counts[SemanticVerdict.AMBIGUOUS],
            invalid_question_count=verdict_counts[SemanticVerdict.INVALID_QUESTION],
            syntactic_accuracy=(syntactic_correct / total) * 100,
            semantic_accuracy=(semantic_correct / total) * 100,
            execution_accuracy=(execution_success / total) * 100,
            table_accuracy=(tables_correct / total) * 100,
            column_accuracy=(columns_correct / total) * 100,
            by_category=by_category,
            by_difficulty=by_difficulty,
            avg_evaluation_time_ms=avg_time,
            total_evaluation_time_ms=total_time,
            timestamp=datetime.now().isoformat(),
            model_used=model_used,
        )

        # Store in history
        self._metrics_history.append(metrics)

        return metrics

    def _get_entry_category(self, entry_id: str) -> str:
        """Get category for an entry ID."""
        entry = self.get_entry(entry_id)
        return entry.category if entry else "unknown"

    def _get_entry_difficulty(self, entry_id: str) -> str:
        """Get difficulty for an entry ID."""
        entry = self.get_entry(entry_id)
        return entry.difficulty if entry else "unknown"

    def get_metrics_history(self) -> List[AccuracyMetrics]:
        """Get historical metrics for trend analysis."""
        return self._metrics_history

    def export_metrics_report(self, metrics: AccuracyMetrics) -> str:
        """
        Generate a human-readable metrics report.

        Args:
            metrics: AccuracyMetrics to report

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "SQL ACCURACY EVALUATION REPORT",
            "=" * 60,
            f"Timestamp: {metrics.timestamp}",
            f"Model: {metrics.model_used}",
            "",
            "OVERALL METRICS",
            "-" * 40,
            f"Total Test Cases: {metrics.total_entries}",
            f"Passed: {metrics.passed} ({metrics.overall_accuracy:.1f}%)",
            f"Failed: {metrics.failed}",
            "",
            "COMPONENT ACCURACY",
            "-" * 40,
            f"Syntactic (valid SQL):  {metrics.syntactic_accuracy:.1f}%",
            f"Semantic (answers Q):   {metrics.semantic_accuracy:.1f}%",
            f"Execution (runs OK):    {metrics.execution_accuracy:.1f}%",
            f"Table Selection:        {metrics.table_accuracy:.1f}%",
            f"Column Selection:       {metrics.column_accuracy:.1f}%",
            "",
            "VERDICT BREAKDOWN",
            "-" * 40,
            f"Correct:          {metrics.correct_count}",
            f"Incorrect:        {metrics.incorrect_count}",
            f"Partial:          {metrics.partial_count}",
            f"Ambiguous:        {metrics.ambiguous_count}",
            f"Invalid Question: {metrics.invalid_question_count}",
        ]

        if metrics.by_difficulty:
            lines.extend([
                "",
                "BY DIFFICULTY",
                "-" * 40,
            ])
            for diff, stats in sorted(metrics.by_difficulty.items()):
                lines.append(
                    f"  {diff.capitalize()}: {stats['passed']}/{stats['total']} "
                    f"({stats['accuracy']:.1f}%)"
                )

        if metrics.by_category:
            lines.extend([
                "",
                "BY CATEGORY",
                "-" * 40,
            ])
            for cat, stats in sorted(metrics.by_category.items()):
                lines.append(
                    f"  {cat.capitalize()}: {stats['passed']}/{stats['total']} "
                    f"({stats['accuracy']:.1f}%)"
                )

        lines.extend([
            "",
            "TIMING",
            "-" * 40,
            f"Average per query: {metrics.avg_evaluation_time_ms:.1f}ms",
            f"Total time: {metrics.total_evaluation_time_ms:.1f}ms",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)


def create_sample_golden_set() -> List[GoldenSetEntry]:
    """
    Create a sample golden set with diverse test cases.

    Returns:
        List of sample GoldenSetEntry objects
    """
    return [
        # Simple queries
        GoldenSetEntry(
            id="simple_001",
            question="How many customers do we have?",
            expected_sql="SELECT COUNT(*) FROM customers",
            expected_tables=["customers"],
            expected_columns=[],
            difficulty="easy",
            category="aggregation",
        ),
        GoldenSetEntry(
            id="simple_002",
            question="List all orders",
            expected_sql="SELECT * FROM orders LIMIT 100",
            expected_tables=["orders"],
            expected_columns=["id", "customer_id", "total"],
            difficulty="easy",
            category="general",
        ),

        # Join queries
        GoldenSetEntry(
            id="join_001",
            question="Show all orders with customer names",
            expected_sql="""
                SELECT o.*, c.name
                FROM orders o
                JOIN customers c ON o.customer_id = c.id
            """,
            expected_tables=["orders", "customers"],
            expected_columns=["name", "customer_id"],
            difficulty="medium",
            category="join",
        ),

        # Aggregation with filters
        GoldenSetEntry(
            id="agg_001",
            question="What is the total revenue by customer?",
            expected_sql="""
                SELECT customer_id, SUM(total) as revenue
                FROM orders
                GROUP BY customer_id
            """,
            expected_tables=["orders"],
            expected_columns=["customer_id", "total"],
            difficulty="medium",
            category="aggregation",
        ),

        # Invalid question - tests detection
        GoldenSetEntry(
            id="invalid_001",
            question="What is the weather forecast for tomorrow?",
            expected_sql="",
            expected_tables=[],
            expected_columns=[],
            difficulty="easy",
            category="invalid",
            is_valid_question=False,
            expected_verdict="invalid_question",
            notes="This question cannot be answered with the database",
        ),

        # Ambiguous question
        GoldenSetEntry(
            id="ambiguous_001",
            question="Show me the data",
            expected_sql="",
            expected_tables=[],
            expected_columns=[],
            difficulty="easy",
            category="ambiguous",
            is_valid_question=True,
            expected_verdict="ambiguous",
            notes="Question is too vague - which data?",
        ),

        # Complex query
        GoldenSetEntry(
            id="complex_001",
            question="Find customers who have placed more than 5 orders with total value over $1000",
            expected_sql="""
                SELECT c.id, c.name, COUNT(o.id) as order_count, SUM(o.total) as total_value
                FROM customers c
                JOIN orders o ON c.id = o.customer_id
                GROUP BY c.id, c.name
                HAVING COUNT(o.id) > 5 AND SUM(o.total) > 1000
            """,
            expected_tables=["customers", "orders"],
            expected_columns=["id", "name", "total"],
            difficulty="hard",
            category="complex",
        ),
    ]
