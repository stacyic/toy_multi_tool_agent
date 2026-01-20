"""Unit tests for HumanReviewQueue and AmbiguousQueryDetector."""

import tempfile
from pathlib import Path

import pytest

from multi_tool_agent.tools.sql_accessor.human_review import (
    AmbiguousQueryDetector,
    HumanReviewQueue,
    ReviewItem,
    ReviewReason,
    ReviewStatus,
)
from multi_tool_agent.tools.sql_accessor.semantic_evaluator import (
    SemanticEvaluation,
    SemanticVerdict,
)


class TestReviewReason:
    """Tests for ReviewReason enum."""

    def test_reason_values(self):
        """Test that all reason values exist."""
        assert ReviewReason.AMBIGUOUS_QUESTION.value == "ambiguous_question"
        assert ReviewReason.INVALID_QUESTION.value == "invalid_question"
        assert ReviewReason.LOW_CONFIDENCE.value == "low_confidence"
        assert ReviewReason.SEMANTIC_MISMATCH.value == "semantic_mismatch"
        assert ReviewReason.REPEATED_FAILURES.value == "repeated_failures"
        assert ReviewReason.USER_REPORTED.value == "user_reported"
        assert ReviewReason.UNCLEAR_INTENT.value == "unclear_intent"


class TestReviewItem:
    """Tests for ReviewItem dataclass."""

    def test_create_item(self):
        """Test creating a review item."""
        item = ReviewItem(
            id="test_001",
            question="Show me the data",
            generated_sql="SELECT * FROM customers",
            reason=ReviewReason.AMBIGUOUS_QUESTION,
        )

        assert item.id == "test_001"
        assert item.status == ReviewStatus.PENDING
        assert item.occurrence_count == 1
        assert item.created_at != ""

    def test_item_with_evaluation(self):
        """Test creating item with semantic evaluation."""
        item = ReviewItem(
            id="test_002",
            question="What is the weather?",
            generated_sql=None,
            reason=ReviewReason.INVALID_QUESTION,
            semantic_evaluation={
                "verdict": "invalid_question",
                "confidence": 0.9,
            },
        )

        assert item.semantic_evaluation is not None
        assert item.semantic_evaluation["verdict"] == "invalid_question"


class TestHumanReviewQueue:
    """Tests for HumanReviewQueue class."""

    @pytest.fixture
    def queue(self):
        """Create a HumanReviewQueue with temp path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "review_queue.json"
            queue = HumanReviewQueue(queue_path=path, max_queue_size=100)
            yield queue

    def test_add_for_review(self, queue):
        """Test adding a question for review."""
        item = queue.add_for_review(
            question="Show me the data",
            reason=ReviewReason.AMBIGUOUS_QUESTION,
            generated_sql="SELECT * FROM customers",
        )

        assert item.id is not None
        assert item.question == "Show me the data"
        assert item.reason == ReviewReason.AMBIGUOUS_QUESTION
        assert len(queue._items) == 1

    def test_add_duplicate_increments_count(self, queue):
        """Test that adding same question increments occurrence count."""
        item1 = queue.add_for_review(
            question="Show me the data",
            reason=ReviewReason.AMBIGUOUS_QUESTION,
        )

        # Add same question again
        item2 = queue.add_for_review(
            question="Show me the data",
            reason=ReviewReason.AMBIGUOUS_QUESTION,
        )

        # Should be same item with incremented count
        assert item1.id == item2.id
        assert item2.occurrence_count == 2
        assert len(queue._items) == 1

    def test_add_similar_question_detected(self, queue):
        """Test that similar questions are grouped."""
        item1 = queue.add_for_review(
            question="show me the data",
            reason=ReviewReason.AMBIGUOUS_QUESTION,
        )

        # Add very similar question (different case)
        item2 = queue.add_for_review(
            question="Show Me The Data",
            reason=ReviewReason.AMBIGUOUS_QUESTION,
        )

        # Should be same item
        assert item1.id == item2.id

    def test_add_with_semantic_evaluation(self, queue):
        """Test adding item with semantic evaluation."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.AMBIGUOUS,
            confidence=0.5,
            reasoning="Question is unclear",
        )

        item = queue.add_for_review(
            question="What about the thing?",
            reason=ReviewReason.AMBIGUOUS_QUESTION,
            semantic_evaluation=evaluation,
        )

        assert item.semantic_evaluation is not None
        assert item.semantic_evaluation["verdict"] == "ambiguous"

    def test_get_pending_items(self, queue):
        """Test getting pending items sorted by occurrence."""
        # Add multiple items with different occurrence counts
        item1 = queue.add_for_review(
            question="Question A",
            reason=ReviewReason.AMBIGUOUS_QUESTION,
        )
        item2 = queue.add_for_review(
            question="Question B",
            reason=ReviewReason.LOW_CONFIDENCE,
        )
        # Increase occurrence of item2
        queue.add_for_review(
            question="Question B",
            reason=ReviewReason.LOW_CONFIDENCE,
        )
        queue.add_for_review(
            question="Question B",
            reason=ReviewReason.LOW_CONFIDENCE,
        )

        pending = queue.get_pending_items()

        # Item B should be first (higher occurrence count)
        assert len(pending) == 2
        assert pending[0].question == "Question B"
        assert pending[0].occurrence_count == 3

    def test_get_items_by_reason(self, queue):
        """Test filtering items by reason."""
        queue.add_for_review(
            question="Question A",
            reason=ReviewReason.AMBIGUOUS_QUESTION,
        )
        queue.add_for_review(
            question="Question B",
            reason=ReviewReason.INVALID_QUESTION,
        )
        queue.add_for_review(
            question="Question C",
            reason=ReviewReason.AMBIGUOUS_QUESTION,
        )

        ambiguous = queue.get_items_by_reason(ReviewReason.AMBIGUOUS_QUESTION)
        assert len(ambiguous) == 2

        invalid = queue.get_items_by_reason(ReviewReason.INVALID_QUESTION)
        assert len(invalid) == 1

    def test_resolve_item(self, queue):
        """Test resolving a review item."""
        item = queue.add_for_review(
            question="Unclear question",
            reason=ReviewReason.AMBIGUOUS_QUESTION,
        )

        result = queue.resolve_item(
            item_id=item.id,
            resolution_notes="User meant to ask about customers",
            resolved_sql="SELECT * FROM customers",
            resolved_by="reviewer@example.com",
        )

        assert result is True

        resolved_item = queue.get_item(item.id)
        assert resolved_item.status == ReviewStatus.RESOLVED
        assert resolved_item.resolved_sql == "SELECT * FROM customers"
        assert resolved_item.resolved_at is not None

    def test_resolve_with_golden_set(self, queue):
        """Test resolving and adding to golden set."""
        item = queue.add_for_review(
            question="Count all customers",
            reason=ReviewReason.LOW_CONFIDENCE,
        )

        queue.resolve_item(
            item_id=item.id,
            resolution_notes="Valid question, adding to golden set",
            resolved_sql="SELECT COUNT(*) FROM customers",
            add_to_golden_set=True,
        )

        resolved_item = queue.get_item(item.id)
        assert resolved_item.status == ReviewStatus.ADDED_TO_GOLDEN_SET

    def test_reject_item(self, queue):
        """Test rejecting a review item."""
        item = queue.add_for_review(
            question="What is the weather?",
            reason=ReviewReason.INVALID_QUESTION,
        )

        result = queue.reject_item(
            item_id=item.id,
            reason="Question cannot be answered with database",
            rejected_by="reviewer@example.com",
        )

        assert result is True

        rejected_item = queue.get_item(item.id)
        assert rejected_item.status == ReviewStatus.REJECTED
        assert "REJECTED" in rejected_item.resolution_notes

    def test_save_and_load(self, queue):
        """Test saving and loading queue."""
        queue.add_for_review(
            question="Question 1",
            reason=ReviewReason.AMBIGUOUS_QUESTION,
        )
        queue.add_for_review(
            question="Question 2",
            reason=ReviewReason.LOW_CONFIDENCE,
        )

        queue.save()

        # Create new queue and load
        new_queue = HumanReviewQueue(queue_path=queue.queue_path)
        new_queue.load()

        assert len(new_queue._items) == 2

    def test_get_statistics(self, queue):
        """Test getting queue statistics."""
        queue.add_for_review("Q1", ReviewReason.AMBIGUOUS_QUESTION)
        queue.add_for_review("Q2", ReviewReason.INVALID_QUESTION)
        queue.add_for_review("Q3", ReviewReason.LOW_CONFIDENCE)

        # Add occurrence to Q1
        queue.add_for_review("Q1", ReviewReason.AMBIGUOUS_QUESTION)

        # Resolve Q2
        item2 = queue.get_items_by_reason(ReviewReason.INVALID_QUESTION)[0]
        queue.resolve_item(item2.id, "Resolved")

        stats = queue.get_statistics()

        assert stats["total_items"] == 3
        assert stats["pending"] == 2
        assert stats["resolved"] == 1
        assert stats["total_occurrences"] == 4  # Q1 has 2
        assert stats["by_reason"]["ambiguous_question"] == 1
        assert stats["by_reason"]["invalid_question"] == 1

    def test_export_for_review(self, queue):
        """Test exporting queue to markdown."""
        queue.add_for_review(
            question="Ambiguous question",
            reason=ReviewReason.AMBIGUOUS_QUESTION,
            generated_sql="SELECT * FROM customers",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "review.md"
            queue.export_for_review(output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "Human Review Queue" in content
            assert "Ambiguous question" in content
            assert "SELECT * FROM customers" in content

    def test_max_queue_size_enforcement(self, queue):
        """Test that queue enforces max size."""
        queue.max_queue_size = 5

        # Add 5 items and resolve 2
        for i in range(5):
            item = queue.add_for_review(f"Question {i}", ReviewReason.AMBIGUOUS_QUESTION)
            if i < 2:
                queue.resolve_item(item.id, "Resolved")

        # Add more items to trigger cleanup
        for i in range(3):
            queue.add_for_review(f"New Question {i}", ReviewReason.LOW_CONFIDENCE)

        # Should have removed oldest resolved items
        assert len(queue._items) <= 6  # Some flexibility in the cleanup


class TestAmbiguousQueryDetector:
    """Tests for AmbiguousQueryDetector class."""

    @pytest.fixture
    def detector(self):
        """Create an AmbiguousQueryDetector."""
        return AmbiguousQueryDetector()

    def test_detect_vague_question(self, detector):
        """Test detection of vague questions."""
        analysis = detector.analyze("Show me the data")

        assert analysis["is_ambiguous"] is True
        assert any("vague" in issue.lower() for issue in analysis["issues"])

    def test_detect_short_question(self, detector):
        """Test detection of too-short questions."""
        analysis = detector.analyze("data?")

        assert analysis["is_ambiguous"] is True
        assert any("short" in issue.lower() for issue in analysis["issues"])

    def test_detect_off_topic_question(self, detector):
        """Test detection of off-topic questions."""
        analysis = detector.analyze("What is the weather forecast for tomorrow?")

        assert analysis["is_ambiguous"] is True
        assert analysis["needs_review"] is True
        assert any("off-topic" in issue.lower() for issue in analysis["issues"])

    def test_clear_question_passes(self, detector):
        """Test that clear questions are not flagged."""
        analysis = detector.analyze(
            "How many customers have placed orders in the last 30 days?"
        )

        # Should not be flagged as needing review
        assert analysis["needs_review"] is False

    def test_specific_words_extracted(self, detector):
        """Test that specific words are identified."""
        analysis = detector.analyze(
            "Show customer orders from electronics category"
        )

        # Should find domain-specific words
        assert "customer" in analysis["specific_words_found"]
        assert "orders" in analysis["specific_words_found"]
        assert "electronics" in analysis["specific_words_found"]

    def test_should_flag_with_semantic_evaluation_ambiguous(self, detector):
        """Test flagging based on semantic evaluation."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.AMBIGUOUS,
            confidence=0.5,
            reasoning="Question is unclear",
        )

        should_flag, reason = detector.should_flag_for_review(
            "Show me something",
            evaluation,
        )

        assert should_flag is True
        assert reason == ReviewReason.AMBIGUOUS_QUESTION

    def test_should_flag_with_semantic_evaluation_invalid(self, detector):
        """Test flagging for invalid questions."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.INVALID_QUESTION,
            confidence=0.9,
            reasoning="Cannot answer with database",
            is_answerable=False,
        )

        should_flag, reason = detector.should_flag_for_review(
            "What is the stock price?",
            evaluation,
        )

        assert should_flag is True
        assert reason == ReviewReason.INVALID_QUESTION

    def test_should_flag_low_confidence(self, detector):
        """Test flagging for low confidence."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.CORRECT,
            confidence=0.4,  # Low confidence
            reasoning="Might be correct",
        )

        should_flag, reason = detector.should_flag_for_review(
            "List customers",
            evaluation,
        )

        assert should_flag is True
        assert reason == ReviewReason.LOW_CONFIDENCE

    def test_should_flag_low_clarity(self, detector):
        """Test flagging for low question clarity."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.CORRECT,
            confidence=0.8,
            reasoning="Query is correct",
            question_clarity_score=0.3,  # Low clarity
        )

        should_flag, reason = detector.should_flag_for_review(
            "Get that stuff",
            evaluation,
        )

        assert should_flag is True
        assert reason == ReviewReason.UNCLEAR_INTENT

    def test_should_not_flag_good_query(self, detector):
        """Test that good queries are not flagged."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.CORRECT,
            confidence=0.95,
            reasoning="Query correctly answers the question",
            question_clarity_score=0.9,
        )

        should_flag, _ = detector.should_flag_for_review(
            "How many customers are there in total?",
            evaluation,
        )

        assert should_flag is False
