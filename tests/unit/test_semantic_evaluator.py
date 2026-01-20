"""Unit tests for SemanticEvaluator."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from multi_tool_agent.tools.sql_accessor.semantic_evaluator import (
    SemanticEvaluator,
    SemanticEvaluation,
    SemanticVerdict,
)


class TestSemanticVerdict:
    """Tests for SemanticVerdict enum."""

    def test_verdict_values(self):
        """Test that all verdict values exist."""
        assert SemanticVerdict.CORRECT.value == "correct"
        assert SemanticVerdict.INCORRECT.value == "incorrect"
        assert SemanticVerdict.PARTIAL.value == "partial"
        assert SemanticVerdict.INVALID_QUESTION.value == "invalid_question"
        assert SemanticVerdict.AMBIGUOUS.value == "ambiguous"


class TestSemanticEvaluation:
    """Tests for SemanticEvaluation dataclass."""

    def test_default_values(self):
        """Test default values for SemanticEvaluation."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.CORRECT,
            confidence=0.9,
            reasoning="Query correctly answers the question",
        )

        assert evaluation.verdict == SemanticVerdict.CORRECT
        assert evaluation.confidence == 0.9
        assert evaluation.issues == []
        assert evaluation.suggestions == []
        assert evaluation.question_clarity_score == 1.0
        assert evaluation.is_answerable is True

    def test_with_issues(self):
        """Test SemanticEvaluation with issues and suggestions."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.PARTIAL,
            confidence=0.6,
            reasoning="Missing some columns",
            issues=["Does not include customer name", "Missing date filter"],
            suggestions=["Add customer name to SELECT", "Add WHERE clause for date"],
            question_clarity_score=0.8,
            is_answerable=True,
        )

        assert evaluation.verdict == SemanticVerdict.PARTIAL
        assert len(evaluation.issues) == 2
        assert len(evaluation.suggestions) == 2


class TestSemanticEvaluator:
    """Tests for SemanticEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create SemanticEvaluator with mock LLM."""
        with patch("multi_tool_agent.tools.sql_accessor.semantic_evaluator.ChatOpenAI"):
            evaluator = SemanticEvaluator(
                model="gpt-4",
                api_key="test-key",
                confidence_threshold=0.7,
            )
            evaluator.set_schema("customers(id, name, email)\norders(id, customer_id, total)")
            return evaluator

    @pytest.fixture
    def mock_correct_response(self):
        """Mock response for correct evaluation."""
        return json.dumps({
            "verdict": "correct",
            "confidence": 0.95,
            "reasoning": "The SQL correctly retrieves customer names as requested",
            "issues": [],
            "suggestions": [],
            "question_clarity_score": 0.9,
            "is_answerable": True,
            "answerable_explanation": "Question is clear and data exists"
        })

    @pytest.fixture
    def mock_incorrect_response(self):
        """Mock response for incorrect evaluation."""
        return json.dumps({
            "verdict": "incorrect",
            "confidence": 0.8,
            "reasoning": "The SQL queries orders instead of customers",
            "issues": ["Wrong table selected", "Does not answer the question"],
            "suggestions": ["Query customers table instead"],
            "question_clarity_score": 0.9,
            "is_answerable": True,
            "answerable_explanation": "Question is answerable but SQL is wrong"
        })

    @pytest.fixture
    def mock_ambiguous_response(self):
        """Mock response for ambiguous question."""
        return json.dumps({
            "verdict": "ambiguous",
            "confidence": 0.5,
            "reasoning": "The question 'show me the data' is too vague",
            "issues": ["Question does not specify which data"],
            "suggestions": ["Ask user to clarify which table or data they want"],
            "question_clarity_score": 0.2,
            "is_answerable": True,
            "answerable_explanation": "Could be answered multiple ways"
        })

    @pytest.fixture
    def mock_invalid_response(self):
        """Mock response for invalid question."""
        return json.dumps({
            "verdict": "invalid_question",
            "confidence": 0.9,
            "reasoning": "Weather data is not available in the database",
            "issues": ["No weather-related tables exist"],
            "suggestions": ["Inform user this data is not available"],
            "question_clarity_score": 0.8,
            "is_answerable": False,
            "answerable_explanation": "Database does not contain weather information"
        })

    @pytest.mark.asyncio
    async def test_evaluate_correct_query(self, evaluator, mock_correct_response):
        """Test evaluation of a semantically correct query."""
        mock_response = MagicMock()
        mock_response.content = mock_correct_response

        with patch.object(
            evaluator, "_llm"
        ) as mock_llm:
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)

            # Need to also mock invoke_with_logging
            with patch(
                "multi_tool_agent.tools.sql_accessor.semantic_evaluator.invoke_with_logging",
                new_callable=AsyncMock,
                return_value=mock_response
            ):
                result = await evaluator.evaluate(
                    question="List all customer names",
                    sql="SELECT name FROM customers",
                )

        assert result.verdict == SemanticVerdict.CORRECT
        assert result.confidence >= 0.9
        assert result.is_answerable is True
        assert len(result.issues) == 0

    @pytest.mark.asyncio
    async def test_evaluate_incorrect_query(self, evaluator, mock_incorrect_response):
        """Test evaluation of a semantically incorrect query."""
        mock_response = MagicMock()
        mock_response.content = mock_incorrect_response

        with patch(
            "multi_tool_agent.tools.sql_accessor.semantic_evaluator.invoke_with_logging",
            new_callable=AsyncMock,
            return_value=mock_response
        ):
            result = await evaluator.evaluate(
                question="List all customer names",
                sql="SELECT * FROM orders",
            )

        assert result.verdict == SemanticVerdict.INCORRECT
        assert len(result.issues) > 0
        assert len(result.suggestions) > 0

    @pytest.mark.asyncio
    async def test_evaluate_ambiguous_question(self, evaluator, mock_ambiguous_response):
        """Test evaluation of an ambiguous question."""
        mock_response = MagicMock()
        mock_response.content = mock_ambiguous_response

        with patch(
            "multi_tool_agent.tools.sql_accessor.semantic_evaluator.invoke_with_logging",
            new_callable=AsyncMock,
            return_value=mock_response
        ):
            result = await evaluator.evaluate(
                question="Show me the data",
                sql="SELECT * FROM customers",
            )

        assert result.verdict == SemanticVerdict.AMBIGUOUS
        assert result.question_clarity_score < 0.5
        assert evaluator.needs_human_review(result) is True

    @pytest.mark.asyncio
    async def test_evaluate_invalid_question(self, evaluator, mock_invalid_response):
        """Test evaluation of an invalid/unanswerable question."""
        mock_response = MagicMock()
        mock_response.content = mock_invalid_response

        with patch(
            "multi_tool_agent.tools.sql_accessor.semantic_evaluator.invoke_with_logging",
            new_callable=AsyncMock,
            return_value=mock_response
        ):
            result = await evaluator.evaluate(
                question="What is the weather forecast?",
                sql="SELECT * FROM customers",
            )

        assert result.verdict == SemanticVerdict.INVALID_QUESTION
        assert result.is_answerable is False
        assert evaluator.needs_human_review(result) is True

    def test_needs_human_review_ambiguous(self, evaluator):
        """Test that ambiguous verdicts need human review."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.AMBIGUOUS,
            confidence=0.5,
            reasoning="Question is unclear",
        )
        assert evaluator.needs_human_review(evaluation) is True

    def test_needs_human_review_invalid_question(self, evaluator):
        """Test that invalid questions need human review."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.INVALID_QUESTION,
            confidence=0.8,
            reasoning="Cannot answer with available data",
            is_answerable=False,
        )
        assert evaluator.needs_human_review(evaluation) is True

    def test_needs_human_review_incorrect(self, evaluator):
        """Test that incorrect verdicts need human review."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.INCORRECT,
            confidence=0.9,
            reasoning="SQL does not answer the question",
        )
        assert evaluator.needs_human_review(evaluation) is True

    def test_needs_human_review_partial(self, evaluator):
        """Test that partial verdicts need human review."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.PARTIAL,
            confidence=0.8,
            reasoning="SQL partially answers the question",
        )
        assert evaluator.needs_human_review(evaluation) is True

    def test_needs_human_review_low_confidence(self, evaluator):
        """Test that low confidence results need human review."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.CORRECT,
            confidence=0.5,  # Below threshold of 0.7
            reasoning="Probably correct",
        )
        assert evaluator.needs_human_review(evaluation) is True

    def test_needs_human_review_correct_high_confidence(self, evaluator):
        """Test that correct, high confidence results don't need review."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.CORRECT,
            confidence=0.95,
            reasoning="Query is definitely correct",
        )
        assert evaluator.needs_human_review(evaluation) is False

    def test_needs_human_review_low_clarity(self, evaluator):
        """Test that low question clarity needs human review."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.CORRECT,
            confidence=0.8,
            reasoning="Query seems correct",
            question_clarity_score=0.3,  # Below 0.5
        )
        assert evaluator.needs_human_review(evaluation) is True

    def test_get_feedback_for_regeneration_correct(self, evaluator):
        """Test that correct queries return no feedback."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.CORRECT,
            confidence=0.9,
            reasoning="Perfect query",
        )
        feedback = evaluator.get_feedback_for_regeneration(evaluation)
        assert feedback is None

    def test_get_feedback_for_regeneration_incorrect(self, evaluator):
        """Test that incorrect queries return detailed feedback."""
        evaluation = SemanticEvaluation(
            verdict=SemanticVerdict.INCORRECT,
            confidence=0.8,
            reasoning="Wrong table used",
            issues=["Used orders instead of customers"],
            suggestions=["Query customers table"],
        )
        feedback = evaluator.get_feedback_for_regeneration(evaluation)

        assert feedback is not None
        assert "incorrect" in feedback.lower()
        assert "Wrong table used" in feedback
        assert "orders instead of customers" in feedback
        assert "customers table" in feedback

    def test_parse_response_with_markdown(self, evaluator):
        """Test parsing response with markdown code blocks."""
        content = """```json
{
    "verdict": "correct",
    "confidence": 0.9,
    "reasoning": "Good query",
    "issues": [],
    "suggestions": [],
    "question_clarity_score": 0.9,
    "is_answerable": true
}
```"""
        result = evaluator._parse_evaluation(content)

        assert result.verdict == SemanticVerdict.CORRECT
        assert result.confidence == 0.9

    def test_parse_response_invalid_json(self, evaluator):
        """Test parsing invalid JSON response."""
        content = "This is not valid JSON"
        result = evaluator._parse_evaluation(content)

        # Should return partial verdict with parsing issue
        assert result.verdict == SemanticVerdict.PARTIAL
        assert result.confidence == 0.5
        assert "parsing failed" in result.issues[0].lower()

    @pytest.mark.asyncio
    async def test_schema_not_set_raises_error(self):
        """Test that evaluation without schema raises error."""
        with patch("multi_tool_agent.tools.sql_accessor.semantic_evaluator.ChatOpenAI"):
            evaluator = SemanticEvaluator(model="gpt-4")
            # Don't call set_schema

            with pytest.raises(ValueError, match="Schema not set"):
                await evaluator.evaluate("question", "sql")
