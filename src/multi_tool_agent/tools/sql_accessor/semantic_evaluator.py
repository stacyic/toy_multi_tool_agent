"""Semantic evaluation of SQL queries using LLM-as-judge.

This module evaluates whether a generated SQL query correctly answers
the user's original question, providing both semantic and structural analysis.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

from langchain_openai import ChatOpenAI

from ...utils.llm_utils import invoke_with_logging

if TYPE_CHECKING:
    from ...logging.trace_logger import TraceLogger


logger = logging.getLogger("multi_tool_agent.semantic_evaluator")


class SemanticVerdict(str, Enum):
    """Verdict from semantic evaluation."""

    CORRECT = "correct"  # SQL correctly answers the question
    INCORRECT = "incorrect"  # SQL does not answer the question
    PARTIAL = "partial"  # SQL partially answers but missing elements
    INVALID_QUESTION = "invalid_question"  # Question cannot be answered with available data
    AMBIGUOUS = "ambiguous"  # Question is unclear and needs clarification


@dataclass
class SemanticEvaluation:
    """Result of semantic evaluation."""

    verdict: SemanticVerdict
    confidence: float  # 0.0 to 1.0
    reasoning: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    question_clarity_score: float = 1.0  # 0.0 (unclear) to 1.0 (very clear)
    is_answerable: bool = True  # Whether question can be answered with available data


class SemanticEvaluator:
    """
    LLM-based semantic evaluator for SQL queries.

    Uses a separate LLM call to evaluate whether the generated SQL
    actually answers the user's question correctly. This provides a
    second layer of validation beyond syntactic correctness.

    Features:
    - Detects semantic mismatches between question and query
    - Identifies invalid/unanswerable questions
    - Flags ambiguous questions for human review
    - Provides actionable feedback for query improvement
    """

    EVALUATION_PROMPT = """You are an expert SQL evaluator. Your task is to determine whether a SQL query correctly answers the user's question.

DATABASE SCHEMA:
{schema}

USER QUESTION:
{question}

GENERATED SQL:
{sql}

EVALUATION CRITERIA:
1. **Semantic Correctness**: Does the SQL retrieve data that reasonably answers the question?
2. **Completeness**: Does the SQL address the main intent of the question?
3. **Question Validity**: Can this question be answered with the available schema?
4. **Reasonable Interpretation**: Accept reasonable interpretations of ambiguous terms.

IMPORTANT - QUESTION CLARITY AND AMBIGUOUS NAMES:
- Questions about specific people using PARTIAL NAMES (e.g., "Alex", "John", "Sarah") are AMBIGUOUS
- Partial names could match multiple people in the database - this is the USER'S responsibility to clarify
- For partial name queries, set question_clarity_score LOW (0.3-0.5) and verdict to "ambiguous"
- The user should provide FULL NAMES (e.g., "Alex Johnson", "John Smith") to avoid ambiguity
- Using LIKE '%name%' for partial names is technically correct but the question itself is ambiguous

IMPORTANT - PII MASKING:
- This system MASKS personally identifiable information (PII) in query results for privacy protection
- PII fields (email, phone, SSN, address, etc.) will appear as masked values like "***@***.com" or "***-***-1234"
- PII masking is INTENTIONAL and CORRECT behavior - do NOT mark queries as incorrect due to masked results
- When evaluating results with PII, the SQL is "correct" if it retrieves the right records, even if PII is masked
- The masking happens AFTER the query executes successfully

IMPORTANT - BE PRACTICAL, NOT PEDANTIC:
- If the question has multiple valid interpretations (e.g., "last year" could mean calendar year or last 365 days), accept ANY reasonable interpretation as "correct"
- Focus on whether the SQL captures the USER'S INTENT, not minor technical details
- Only mark as "incorrect" if the SQL fundamentally fails to answer the question (wrong tables, wrong aggregation, missing key filters)
- Common phrases should be interpreted reasonably:
  - "last year" / "last month" = either calendar period OR rolling period - both are correct
  - "top customers" = by revenue, order count, or other reasonable metric - all are correct
  - "recent orders" = any reasonable time window is acceptable

OUTPUT FORMAT (JSON):
{{
    "verdict": "correct" | "incorrect" | "partial" | "invalid_question" | "ambiguous",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of your evaluation",
    "issues": ["List of specific issues found, if any"],
    "suggestions": ["Suggestions for improving the query or clarifying the question"],
    "question_clarity_score": 0.0-1.0,
    "is_answerable": true | false,
    "answerable_explanation": "Why the question is or isn't answerable"
}}

VERDICT GUIDELINES:
- "correct": SQL reasonably answers the question (use this for reasonable interpretations)
- "incorrect": SQL fundamentally fails to answer the question (wrong data, wrong logic)
- "partial": SQL answers part of the question but misses a KEY element explicitly asked for
- "invalid_question": Question cannot be answered with available data
- "ambiguous": Question uses partial/ambiguous identifiers (like first names only) that could match multiple records

QUESTION CLARITY SCORING:
- 1.0: Clear, unambiguous question with specific identifiers
- 0.7-0.9: Mostly clear but minor ambiguity
- 0.4-0.6: Ambiguous (e.g., partial names, vague time references)
- 0.1-0.3: Very unclear, multiple interpretations possible
- 0.0: Cannot understand the question

Only output valid JSON. Do not include any other text."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        logger: Optional["TraceLogger"] = None,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize the semantic evaluator.

        Args:
            model: OpenAI model for evaluation (can differ from generator)
            api_key: OpenAI API key
            temperature: LLM temperature (0 for consistent evaluation)
            logger: Optional TraceLogger for API call logging
            confidence_threshold: Minimum confidence to accept as correct
        """
        self.model = model
        self.trace_logger = logger
        self.confidence_threshold = confidence_threshold

        kwargs = {"model": model, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        self._llm = ChatOpenAI(**kwargs)
        self._schema_description: Optional[str] = None

    def set_schema(self, schema_description: str) -> None:
        """Set the database schema description."""
        self._schema_description = schema_description

    async def evaluate(
        self,
        question: str,
        sql: str,
        execution_result: Optional[str] = None,
    ) -> SemanticEvaluation:
        """
        Evaluate whether SQL correctly answers the question.

        Args:
            question: Original user question
            sql: Generated SQL query
            execution_result: Optional result from SQL execution (for context)

        Returns:
            SemanticEvaluation with verdict and detailed feedback
        """
        if not self._schema_description:
            raise ValueError("Schema not set. Call set_schema() first.")

        # Build evaluation prompt
        prompt = self.EVALUATION_PROMPT.format(
            schema=self._schema_description,
            question=question,
            sql=sql,
        )

        # Add execution result context if available
        if execution_result:
            prompt += f"\n\nEXECUTION RESULT (first few rows):\n{execution_result[:500]}"

        messages = [{"role": "user", "content": prompt}]

        try:
            response = await invoke_with_logging(
                llm=self._llm,
                messages=messages,
                logger=self.trace_logger,
                component="semantic_evaluation",
                model=self.model,
            )

            return self._parse_evaluation(response.content)

        except Exception as e:
            logger.error(f"Semantic evaluation failed: {e}")
            # Return uncertain evaluation on failure
            return SemanticEvaluation(
                verdict=SemanticVerdict.PARTIAL,
                confidence=0.5,
                reasoning=f"Evaluation could not be completed: {str(e)}",
                issues=["Evaluation process failed"],
                suggestions=["Manual review recommended"],
            )

    def _parse_evaluation(self, content: str) -> SemanticEvaluation:
        """Parse LLM response into SemanticEvaluation."""
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
            if match:
                content = match.group(1).strip()

        try:
            data = json.loads(content)

            verdict_str = data.get("verdict", "partial").lower()
            verdict_map = {
                "correct": SemanticVerdict.CORRECT,
                "incorrect": SemanticVerdict.INCORRECT,
                "partial": SemanticVerdict.PARTIAL,
                "invalid_question": SemanticVerdict.INVALID_QUESTION,
                "ambiguous": SemanticVerdict.AMBIGUOUS,
            }
            verdict = verdict_map.get(verdict_str, SemanticVerdict.PARTIAL)

            return SemanticEvaluation(
                verdict=verdict,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "No reasoning provided"),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                question_clarity_score=float(data.get("question_clarity_score", 1.0)),
                is_answerable=data.get("is_answerable", True),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse evaluation response: {e}")
            return SemanticEvaluation(
                verdict=SemanticVerdict.PARTIAL,
                confidence=0.5,
                reasoning=f"Could not parse evaluation: {content[:200]}",
                issues=["Evaluation parsing failed"],
                suggestions=["Manual review recommended"],
            )

    def needs_human_review(self, evaluation: SemanticEvaluation) -> bool:
        """
        Determine if the evaluation result requires human review.

        Automatically flags any result that is not 'correct' for human review.

        Args:
            evaluation: SemanticEvaluation result

        Returns:
            True if human review is recommended (any non-correct result)
        """
        # Flag for human review if result is anything other than 'correct'
        # This includes: incorrect, partial, ambiguous, invalid_question
        # Also flag if confidence is low even for 'correct' verdicts
        return (
            evaluation.verdict != SemanticVerdict.CORRECT
            or evaluation.confidence < self.confidence_threshold
            or evaluation.question_clarity_score < 0.5
        )

    def get_feedback_for_regeneration(
        self, evaluation: SemanticEvaluation
    ) -> Optional[str]:
        """
        Generate feedback for SQL regeneration based on evaluation.

        Args:
            evaluation: SemanticEvaluation result

        Returns:
            Feedback string for query regeneration, or None if query is correct
        """
        if evaluation.verdict == SemanticVerdict.CORRECT:
            return None

        feedback_parts = [f"Semantic evaluation: {evaluation.verdict.value}"]
        feedback_parts.append(f"Reasoning: {evaluation.reasoning}")

        if evaluation.issues:
            feedback_parts.append("Issues found:")
            for issue in evaluation.issues:
                feedback_parts.append(f"  - {issue}")

        if evaluation.suggestions:
            feedback_parts.append("Suggestions:")
            for suggestion in evaluation.suggestions:
                feedback_parts.append(f"  - {suggestion}")

        return "\n".join(feedback_parts)
