"""Accuracy metrics tracking and evaluation runner.

This module provides infrastructure for running evaluations against
the golden set and tracking accuracy metrics over time.
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .golden_set import (
    AccuracyMetrics,
    EvaluationResult,
    GoldenSetEntry,
    GoldenSetManager,
)
from .semantic_evaluator import SemanticEvaluation, SemanticEvaluator, SemanticVerdict

if TYPE_CHECKING:
    from .query_checker import QueryChecker
    from .query_executor import QueryExecutor
    from .query_generator import QueryGenerator


logger = logging.getLogger("multi_tool_agent.accuracy_tracker")


@dataclass
class RunConfig:
    """Configuration for an evaluation run."""

    run_id: str
    model: str
    timestamp: str
    categories: Optional[List[str]] = None  # None = all categories
    difficulties: Optional[List[str]] = None  # None = all difficulties
    sample_size: Optional[int] = None  # None = full golden set


class AccuracyTracker:
    """
    Runs evaluations and tracks accuracy metrics over time.

    Features:
    - Run full or partial golden set evaluations
    - Track metrics history for trend analysis
    - Compare performance across models
    - Export detailed reports
    """

    def __init__(
        self,
        golden_set_manager: GoldenSetManager,
        generator: "QueryGenerator",
        checker: "QueryChecker",
        executor: "QueryExecutor",
        semantic_evaluator: SemanticEvaluator,
        metrics_dir: Optional[Path] = None,
    ):
        """
        Initialize the accuracy tracker.

        Args:
            golden_set_manager: Manager for golden set data
            generator: SQL query generator to evaluate
            checker: SQL query checker for syntactic validation
            executor: SQL executor for running queries
            semantic_evaluator: Evaluator for semantic correctness
            metrics_dir: Directory to store metrics history
        """
        self.golden_set = golden_set_manager
        self.generator = generator
        self.checker = checker
        self.executor = executor
        self.semantic_evaluator = semantic_evaluator
        self.metrics_dir = metrics_dir or Path("data/metrics")
        self._run_history: List[Tuple[RunConfig, AccuracyMetrics]] = []

    async def run_evaluation(
        self,
        run_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
        save_results: bool = True,
    ) -> Tuple[List[EvaluationResult], AccuracyMetrics]:
        """
        Run evaluation against the golden set.

        Args:
            run_id: Optional identifier for this run
            categories: Filter by categories (None = all)
            difficulties: Filter by difficulties (None = all)
            sample_size: Limit number of entries (None = all)
            save_results: Whether to save results to disk

        Returns:
            Tuple of (detailed results, aggregated metrics)
        """
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        model = getattr(self.generator, 'model', 'unknown')

        config = RunConfig(
            run_id=run_id,
            model=model,
            timestamp=datetime.now().isoformat(),
            categories=categories,
            difficulties=difficulties,
            sample_size=sample_size,
        )

        # Get entries to evaluate
        entries = self._filter_entries(categories, difficulties, sample_size)

        if not entries:
            logger.warning("No entries to evaluate after filtering")
            return [], AccuracyMetrics(
                total_entries=0,
                passed=0,
                failed=0,
                timestamp=config.timestamp,
                model_used=model,
            )

        logger.info(f"Starting evaluation run {run_id} with {len(entries)} entries")

        # Run evaluation on each entry
        results = []
        for entry in entries:
            result = await self._evaluate_entry(entry)
            results.append(result)

            # Log progress
            status = "PASS" if result.overall_pass else "FAIL"
            logger.info(f"  [{status}] {entry.id}: {entry.question[:50]}...")

        # Calculate metrics
        metrics = self.golden_set.calculate_metrics(results, model)

        # Store in history
        self._run_history.append((config, metrics))

        # Save results if requested
        if save_results:
            self._save_run_results(config, results, metrics)

        logger.info(
            f"Evaluation complete: {metrics.passed}/{metrics.total_entries} passed "
            f"({metrics.overall_accuracy:.1f}%)"
        )

        return results, metrics

    def _filter_entries(
        self,
        categories: Optional[List[str]],
        difficulties: Optional[List[str]],
        sample_size: Optional[int],
    ) -> List[GoldenSetEntry]:
        """Filter golden set entries based on criteria."""
        entries = self.golden_set.entries

        if categories:
            entries = [e for e in entries if e.category in categories]

        if difficulties:
            entries = [e for e in entries if e.difficulty in difficulties]

        if sample_size and sample_size < len(entries):
            # Random sample
            import random
            entries = random.sample(entries, sample_size)

        return entries

    async def _evaluate_entry(self, entry: GoldenSetEntry) -> EvaluationResult:
        """Evaluate a single golden set entry."""
        start_time = time.time()

        # Default values
        generated_sql = ""
        syntactic_valid = False
        tables_match = False
        columns_match = False
        execution_success = False
        execution_match = False
        semantic_evaluation: Optional[SemanticEvaluation] = None
        error_message = None

        try:
            # Step 1: Generate SQL
            gen_result = await self.generator.generate(
                question=entry.question,
            )
            generated_sql = gen_result.sql

            # Step 2: Syntactic validation
            validation = self.checker.validate(generated_sql)
            syntactic_valid = validation.is_valid

            # Step 3: Check table/column coverage
            if syntactic_valid:
                tables_match = self._check_tables_match(
                    validation.tables_used, set(entry.expected_tables)
                )
                columns_match = self._check_columns_match(
                    validation.columns_used, set(entry.expected_columns)
                )

            # Step 4: Execute query
            if syntactic_valid:
                exec_result = await self.executor.execute(generated_sql)
                execution_success = exec_result.success

                if exec_result.success:
                    # Format result for semantic evaluation
                    result_preview = self._format_result_preview(
                        exec_result.columns, exec_result.rows[:5]
                    )

                    # Step 5: Semantic evaluation
                    semantic_evaluation = await self.semantic_evaluator.evaluate(
                        question=entry.question,
                        sql=generated_sql,
                        execution_result=result_preview,
                    )
                else:
                    error_message = exec_result.error
            else:
                error_message = "; ".join(validation.errors)

        except Exception as e:
            error_message = str(e)
            logger.error(f"Error evaluating entry {entry.id}: {e}")

        # Determine overall pass/fail
        if semantic_evaluation:
            # For invalid questions, pass if we correctly identified them
            if not entry.is_valid_question:
                overall_pass = (
                    semantic_evaluation.verdict == SemanticVerdict.INVALID_QUESTION
                    or not semantic_evaluation.is_answerable
                )
            # For valid questions, pass if semantic evaluation is correct
            else:
                overall_pass = semantic_evaluation.verdict == SemanticVerdict.CORRECT
        else:
            overall_pass = False

        evaluation_time = (time.time() - start_time) * 1000

        return EvaluationResult(
            entry_id=entry.id,
            question=entry.question,
            expected_sql=entry.expected_sql,
            generated_sql=generated_sql,
            semantic_verdict=semantic_evaluation.verdict
            if semantic_evaluation else SemanticVerdict.INCORRECT,
            semantic_confidence=semantic_evaluation.confidence
            if semantic_evaluation else 0.0,
            syntactic_valid=syntactic_valid,
            tables_match=tables_match,
            columns_match=columns_match,
            execution_success=execution_success,
            execution_match=execution_match,
            overall_pass=overall_pass,
            error_message=error_message,
            evaluation_time_ms=evaluation_time,
        )

    def _check_tables_match(
        self, actual: set, expected: set
    ) -> bool:
        """Check if actual tables cover expected tables."""
        if not expected:
            return True
        actual_lower = {t.lower() for t in actual}
        expected_lower = {t.lower() for t in expected}
        return expected_lower.issubset(actual_lower)

    def _check_columns_match(
        self, actual: set, expected: set
    ) -> bool:
        """Check if actual columns cover expected columns."""
        if not expected:
            return True
        actual_lower = {c.lower() for c in actual}
        expected_lower = {c.lower() for c in expected}
        return expected_lower.issubset(actual_lower)

    def _format_result_preview(
        self, columns: List[str], rows: List[tuple]
    ) -> str:
        """Format SQL result for semantic evaluation context."""
        if not rows:
            return "No results returned"

        lines = [" | ".join(columns)]
        lines.append("-" * 40)
        for row in rows:
            lines.append(" | ".join(str(v) for v in row))

        return "\n".join(lines)

    def _save_run_results(
        self,
        config: RunConfig,
        results: List[EvaluationResult],
        metrics: AccuracyMetrics,
    ) -> None:
        """Save evaluation results to disk."""
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_file = self.metrics_dir / f"run_{config.run_id}_results.json"
        results_data = {
            "config": asdict(config),
            "results": [asdict(r) for r in results],
        }
        # Convert enums to strings
        for r in results_data["results"]:
            r["semantic_verdict"] = r["semantic_verdict"].value

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        # Save metrics summary
        metrics_file = self.metrics_dir / f"run_{config.run_id}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(asdict(metrics), f, indent=2, default=str)

        # Append to history file
        history_file = self.metrics_dir / "metrics_history.jsonl"
        history_entry = {
            "run_id": config.run_id,
            "timestamp": config.timestamp,
            "model": config.model,
            "overall_accuracy": metrics.overall_accuracy,
            "semantic_accuracy": metrics.semantic_accuracy,
            "syntactic_accuracy": metrics.syntactic_accuracy,
            "total_entries": metrics.total_entries,
        }
        with open(history_file, "a") as f:
            f.write(json.dumps(history_entry) + "\n")

        logger.info(f"Saved results to {self.metrics_dir}")

    def get_trend_report(self, last_n_runs: int = 10) -> str:
        """
        Generate a trend report from recent runs.

        Args:
            last_n_runs: Number of recent runs to include

        Returns:
            Formatted trend report
        """
        # Load history from file if available
        history_file = self.metrics_dir / "metrics_history.jsonl"
        history = []

        if history_file.exists():
            with open(history_file, "r") as f:
                for line in f:
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        recent = history[-last_n_runs:] if history else []

        if not recent:
            return "No evaluation history available."

        lines = [
            "=" * 60,
            "ACCURACY TREND REPORT",
            "=" * 60,
            "",
            f"Showing last {len(recent)} runs:",
            "",
            f"{'Run ID':<20} {'Date':<12} {'Model':<15} {'Accuracy':>10}",
            "-" * 60,
        ]

        for entry in recent:
            run_id = entry.get("run_id", "N/A")[:18]
            timestamp = entry.get("timestamp", "")[:10]
            model = entry.get("model", "unknown")[:13]
            accuracy = entry.get("overall_accuracy", 0)
            lines.append(f"{run_id:<20} {timestamp:<12} {model:<15} {accuracy:>9.1f}%")

        # Calculate trend
        if len(recent) >= 2:
            first_acc = recent[0].get("overall_accuracy", 0)
            last_acc = recent[-1].get("overall_accuracy", 0)
            change = last_acc - first_acc
            direction = "+" if change >= 0 else ""
            lines.extend([
                "",
                "-" * 60,
                f"Trend: {direction}{change:.1f}% from first to last run",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)

    def compare_models(
        self, model_runs: Dict[str, str]
    ) -> str:
        """
        Compare accuracy across different model runs.

        Args:
            model_runs: Dict mapping model name to run_id

        Returns:
            Comparison report
        """
        lines = [
            "=" * 60,
            "MODEL COMPARISON REPORT",
            "=" * 60,
            "",
        ]

        metrics_by_model = {}
        for model, run_id in model_runs.items():
            metrics_file = self.metrics_dir / f"run_{run_id}_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics_by_model[model] = json.load(f)

        if not metrics_by_model:
            return "No metrics found for specified runs."

        headers = ["Metric"] + list(metrics_by_model.keys())
        lines.append(" | ".join(f"{h:<15}" for h in headers))
        lines.append("-" * (17 * len(headers)))

        metric_keys = [
            ("overall_accuracy", "Overall"),
            ("semantic_accuracy", "Semantic"),
            ("syntactic_accuracy", "Syntactic"),
            ("execution_accuracy", "Execution"),
        ]

        for key, label in metric_keys:
            values = [label]
            for model in model_runs.keys():
                m = metrics_by_model.get(model, {})
                val = m.get(key, 0)
                values.append(f"{val:.1f}%")
            lines.append(" | ".join(f"{v:<15}" for v in values))

        lines.append("=" * 60)
        return "\n".join(lines)


async def run_quick_evaluation(
    sql_accessor,
    questions: List[str],
) -> Dict[str, Any]:
    """
    Run a quick evaluation on a list of questions without golden set.

    Useful for spot-checking during development.

    Args:
        sql_accessor: Configured SQLAccessor instance
        questions: List of questions to evaluate

    Returns:
        Summary of results
    """
    results = []

    for question in questions:
        try:
            result = await sql_accessor.run(question)
            results.append({
                "question": question,
                "success": result.success,
                "sql": result.sql_executed,
                "error": result.error,
            })
        except Exception as e:
            results.append({
                "question": question,
                "success": False,
                "error": str(e),
            })

    success_count = sum(1 for r in results if r["success"])

    return {
        "total": len(questions),
        "success": success_count,
        "failed": len(questions) - success_count,
        "accuracy": (success_count / len(questions)) * 100 if questions else 0,
        "results": results,
    }
