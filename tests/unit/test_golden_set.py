"""Unit tests for GoldenSetManager and related components."""

import json
import tempfile
from pathlib import Path

import pytest

from multi_tool_agent.tools.sql_accessor.golden_set import (
    AccuracyMetrics,
    EvaluationResult,
    GoldenSetEntry,
    GoldenSetManager,
    create_sample_golden_set,
)
from multi_tool_agent.tools.sql_accessor.semantic_evaluator import SemanticVerdict


class TestGoldenSetEntry:
    """Tests for GoldenSetEntry dataclass."""

    def test_create_entry(self):
        """Test creating a golden set entry."""
        entry = GoldenSetEntry(
            id="test_001",
            question="How many customers?",
            expected_sql="SELECT COUNT(*) FROM customers",
            expected_tables=["customers"],
            expected_columns=[],
        )

        assert entry.id == "test_001"
        assert entry.difficulty == "medium"  # default
        assert entry.category == "general"  # default
        assert entry.is_valid_question is True  # default

    def test_create_invalid_entry(self):
        """Test creating an entry for invalid question."""
        entry = GoldenSetEntry(
            id="invalid_001",
            question="What is the weather?",
            expected_sql="",
            expected_tables=[],
            expected_columns=[],
            is_valid_question=False,
            expected_verdict="invalid_question",
        )

        assert entry.is_valid_question is False
        assert entry.expected_verdict == "invalid_question"


class TestGoldenSetManager:
    """Tests for GoldenSetManager class."""

    @pytest.fixture
    def manager(self):
        """Create a GoldenSetManager with temp path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "golden_set.json"
            manager = GoldenSetManager(golden_set_path=path)
            yield manager

    @pytest.fixture
    def sample_entry(self):
        """Create a sample golden set entry."""
        return GoldenSetEntry(
            id="sample_001",
            question="List all customers",
            expected_sql="SELECT * FROM customers LIMIT 100",
            expected_tables=["customers"],
            expected_columns=["id", "name", "email"],
            difficulty="easy",
            category="general",
        )

    def test_add_entry(self, manager, sample_entry):
        """Test adding an entry to the golden set."""
        manager.add_entry(sample_entry)

        assert len(manager.entries) == 1
        assert manager.get_entry("sample_001") is not None

    def test_add_duplicate_entry_raises(self, manager, sample_entry):
        """Test that adding duplicate ID raises error."""
        manager.add_entry(sample_entry)

        with pytest.raises(ValueError, match="already exists"):
            manager.add_entry(sample_entry)

    def test_remove_entry(self, manager, sample_entry):
        """Test removing an entry."""
        manager.add_entry(sample_entry)
        result = manager.remove_entry("sample_001")

        assert result is True
        assert len(manager.entries) == 0

    def test_remove_nonexistent_entry(self, manager):
        """Test removing non-existent entry returns False."""
        result = manager.remove_entry("nonexistent")
        assert result is False

    def test_get_entry(self, manager, sample_entry):
        """Test getting an entry by ID."""
        manager.add_entry(sample_entry)
        entry = manager.get_entry("sample_001")

        assert entry is not None
        assert entry.question == "List all customers"

    def test_get_entries_by_category(self, manager):
        """Test filtering entries by category."""
        entry1 = GoldenSetEntry(
            id="agg_001",
            question="Count customers",
            expected_sql="SELECT COUNT(*) FROM customers",
            expected_tables=["customers"],
            expected_columns=[],
            category="aggregation",
        )
        entry2 = GoldenSetEntry(
            id="join_001",
            question="Show orders with customers",
            expected_sql="SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id",
            expected_tables=["orders", "customers"],
            expected_columns=[],
            category="join",
        )

        manager.add_entry(entry1)
        manager.add_entry(entry2)

        agg_entries = manager.get_entries_by_category("aggregation")
        assert len(agg_entries) == 1
        assert agg_entries[0].id == "agg_001"

    def test_get_entries_by_difficulty(self, manager):
        """Test filtering entries by difficulty."""
        entry1 = GoldenSetEntry(
            id="easy_001",
            question="List customers",
            expected_sql="SELECT * FROM customers",
            expected_tables=["customers"],
            expected_columns=[],
            difficulty="easy",
        )
        entry2 = GoldenSetEntry(
            id="hard_001",
            question="Complex query",
            expected_sql="SELECT ...",
            expected_tables=["customers", "orders"],
            expected_columns=[],
            difficulty="hard",
        )

        manager.add_entry(entry1)
        manager.add_entry(entry2)

        easy_entries = manager.get_entries_by_difficulty("easy")
        assert len(easy_entries) == 1
        assert easy_entries[0].id == "easy_001"

    def test_save_and_load(self, manager, sample_entry):
        """Test saving and loading golden set."""
        manager.add_entry(sample_entry)
        manager.save()

        # Create new manager and load
        new_manager = GoldenSetManager(golden_set_path=manager.golden_set_path)
        new_manager.load()

        assert len(new_manager.entries) == 1
        loaded_entry = new_manager.get_entry("sample_001")
        assert loaded_entry.question == sample_entry.question

    def test_load_nonexistent_file(self, manager):
        """Test loading from non-existent file starts empty."""
        manager.load()
        assert len(manager.entries) == 0


class TestAccuracyMetrics:
    """Tests for AccuracyMetrics calculations."""

    @pytest.fixture
    def manager(self):
        """Create a GoldenSetManager with sample entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "golden_set.json"
            manager = GoldenSetManager(golden_set_path=path)

            # Add entries for different categories/difficulties
            for i, (cat, diff) in enumerate([
                ("general", "easy"),
                ("general", "medium"),
                ("aggregation", "easy"),
                ("join", "hard"),
            ]):
                manager.add_entry(GoldenSetEntry(
                    id=f"test_{i:03d}",
                    question=f"Question {i}",
                    expected_sql=f"SELECT * FROM table_{i}",
                    expected_tables=[f"table_{i}"],
                    expected_columns=[],
                    category=cat,
                    difficulty=diff,
                ))

            yield manager

    @pytest.fixture
    def sample_results(self):
        """Create sample evaluation results."""
        return [
            EvaluationResult(
                entry_id="test_000",
                question="Question 0",
                expected_sql="SELECT * FROM table_0",
                generated_sql="SELECT * FROM table_0",
                semantic_verdict=SemanticVerdict.CORRECT,
                semantic_confidence=0.95,
                syntactic_valid=True,
                tables_match=True,
                columns_match=True,
                execution_success=True,
                execution_match=True,
                overall_pass=True,
                evaluation_time_ms=100,
            ),
            EvaluationResult(
                entry_id="test_001",
                question="Question 1",
                expected_sql="SELECT * FROM table_1",
                generated_sql="SELECT * FROM table_1",
                semantic_verdict=SemanticVerdict.CORRECT,
                semantic_confidence=0.9,
                syntactic_valid=True,
                tables_match=True,
                columns_match=True,
                execution_success=True,
                execution_match=True,
                overall_pass=True,
                evaluation_time_ms=120,
            ),
            EvaluationResult(
                entry_id="test_002",
                question="Question 2",
                expected_sql="SELECT * FROM table_2",
                generated_sql="SELECT * FROM wrong_table",
                semantic_verdict=SemanticVerdict.INCORRECT,
                semantic_confidence=0.8,
                syntactic_valid=True,
                tables_match=False,
                columns_match=True,
                execution_success=True,
                execution_match=False,
                overall_pass=False,
                evaluation_time_ms=150,
            ),
            EvaluationResult(
                entry_id="test_003",
                question="Question 3",
                expected_sql="SELECT * FROM table_3",
                generated_sql="INVALID SQL",
                semantic_verdict=SemanticVerdict.INCORRECT,
                semantic_confidence=0.5,
                syntactic_valid=False,
                tables_match=False,
                columns_match=False,
                execution_success=False,
                execution_match=False,
                overall_pass=False,
                error_message="Syntax error",
                evaluation_time_ms=50,
            ),
        ]

    def test_calculate_metrics(self, manager, sample_results):
        """Test calculating metrics from results."""
        metrics = manager.calculate_metrics(sample_results, model_used="gpt-4")

        assert metrics.total_entries == 4
        assert metrics.passed == 2
        assert metrics.failed == 2
        assert metrics.overall_accuracy == 50.0

    def test_component_accuracies(self, manager, sample_results):
        """Test component-level accuracy calculations."""
        metrics = manager.calculate_metrics(sample_results)

        # 3 out of 4 have valid syntax
        assert metrics.syntactic_accuracy == 75.0
        # 2 out of 4 are semantically correct
        assert metrics.semantic_accuracy == 50.0
        # 3 out of 4 executed successfully
        assert metrics.execution_accuracy == 75.0

    def test_verdict_breakdown(self, manager, sample_results):
        """Test verdict count breakdown."""
        metrics = manager.calculate_metrics(sample_results)

        assert metrics.correct_count == 2
        assert metrics.incorrect_count == 2
        assert metrics.partial_count == 0
        assert metrics.ambiguous_count == 0

    def test_timing_metrics(self, manager, sample_results):
        """Test timing metric calculations."""
        metrics = manager.calculate_metrics(sample_results)

        # Average of [100, 120, 150, 50] = 105
        assert metrics.avg_evaluation_time_ms == 105.0
        assert metrics.total_evaluation_time_ms == 420.0

    def test_empty_results(self, manager):
        """Test metrics with empty results."""
        metrics = manager.calculate_metrics([])

        assert metrics.total_entries == 0
        assert metrics.overall_accuracy == 0.0

    def test_export_metrics_report(self, manager, sample_results):
        """Test generating human-readable report."""
        metrics = manager.calculate_metrics(sample_results, model_used="gpt-4")
        report = manager.export_metrics_report(metrics)

        assert "SQL ACCURACY EVALUATION REPORT" in report
        assert "gpt-4" in report
        assert "50.0%" in report  # Overall accuracy
        assert "COMPONENT ACCURACY" in report
        assert "VERDICT BREAKDOWN" in report


class TestSampleGoldenSet:
    """Tests for sample golden set creation."""

    def test_create_sample_golden_set(self):
        """Test that sample golden set creates valid entries."""
        entries = create_sample_golden_set()

        assert len(entries) > 0

        # Check for variety
        categories = {e.category for e in entries}
        difficulties = {e.difficulty for e in entries}

        assert "aggregation" in categories
        assert "join" in categories
        assert "invalid" in categories
        assert "ambiguous" in categories

        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_sample_set_has_invalid_questions(self):
        """Test that sample set includes invalid question examples."""
        entries = create_sample_golden_set()
        invalid_entries = [e for e in entries if not e.is_valid_question]

        assert len(invalid_entries) >= 1

    def test_sample_set_has_ambiguous_questions(self):
        """Test that sample set includes ambiguous question examples."""
        entries = create_sample_golden_set()
        ambiguous_entries = [e for e in entries if e.category == "ambiguous"]

        assert len(ambiguous_entries) >= 1
