"""Human review queue for ambiguous and problematic queries.

This module provides infrastructure for recording queries that require
human review, managing a review queue, and tracking resolution status.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .semantic_evaluator import SemanticEvaluation, SemanticVerdict


logger = logging.getLogger("multi_tool_agent.human_review")


class ReviewReason(str, Enum):
    """Reasons why a query needs human review."""

    AMBIGUOUS_QUESTION = "ambiguous_question"
    INVALID_QUESTION = "invalid_question"
    LOW_CONFIDENCE = "low_confidence"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    REPEATED_FAILURES = "repeated_failures"
    USER_REPORTED = "user_reported"
    UNCLEAR_INTENT = "unclear_intent"


class ReviewStatus(str, Enum):
    """Status of a review item."""

    PENDING = "pending"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"
    REJECTED = "rejected"  # Question is intentionally unanswerable
    ADDED_TO_GOLDEN_SET = "added_to_golden_set"


@dataclass
class ReviewItem:
    """A single item in the human review queue."""

    id: str
    question: str
    generated_sql: Optional[str]
    reason: ReviewReason
    status: ReviewStatus = ReviewStatus.PENDING

    # Context
    semantic_evaluation: Optional[Dict[str, Any]] = None
    execution_result: Optional[str] = None
    error_message: Optional[str] = None

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    session_id: Optional[str] = None
    user_context: Optional[str] = None  # Additional context from user

    # Resolution
    resolution_notes: Optional[str] = None
    resolved_sql: Optional[str] = None  # Correct SQL if determined
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None

    # For tracking patterns
    similar_questions: List[str] = field(default_factory=list)
    occurrence_count: int = 1

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


class HumanReviewQueue:
    """
    Manages a queue of queries requiring human review.

    Features:
    - Automatic detection and recording of problematic queries
    - Deduplication of similar questions
    - Priority-based review ordering
    - Resolution tracking
    - Export for review tools or dashboards
    """

    def __init__(
        self,
        queue_path: Optional[Path] = None,
        max_queue_size: int = 1000,
        similarity_threshold: float = 0.8,
    ):
        """
        Initialize the human review queue.

        Args:
            queue_path: Path to persist queue (JSON file)
            max_queue_size: Maximum items to keep in queue
            similarity_threshold: Threshold for considering questions similar
        """
        self.queue_path = queue_path or Path("data/review_queue.json")
        self.max_queue_size = max_queue_size
        self.similarity_threshold = similarity_threshold
        self._items: Dict[str, ReviewItem] = {}
        self._question_index: Dict[str, str] = {}  # question -> item_id for dedup

    def load(self, path: Optional[Path] = None) -> None:
        """Load queue from file."""
        load_path = path or self.queue_path

        if not load_path.exists():
            logger.info(f"Review queue file not found at {load_path}, starting empty")
            return

        try:
            with open(load_path, "r") as f:
                data = json.load(f)

            for item_data in data.get("items", []):
                # Convert string enums back
                item_data["reason"] = ReviewReason(item_data["reason"])
                item_data["status"] = ReviewStatus(item_data["status"])
                item = ReviewItem(**item_data)
                self._items[item.id] = item
                self._question_index[item.question.lower().strip()] = item.id

            logger.info(f"Loaded {len(self._items)} review items")

        except Exception as e:
            logger.error(f"Failed to load review queue: {e}")

    def save(self, path: Optional[Path] = None) -> None:
        """Save queue to file."""
        save_path = path or self.queue_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "items": [asdict(item) for item in self._items.values()],
        }

        # Convert enums to strings for JSON
        for item in data["items"]:
            item["reason"] = item["reason"].value
            item["status"] = item["status"].value

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved {len(self._items)} review items to {save_path}")

    def add_for_review(
        self,
        question: str,
        reason: ReviewReason,
        generated_sql: Optional[str] = None,
        semantic_evaluation: Optional[SemanticEvaluation] = None,
        execution_result: Optional[str] = None,
        error_message: Optional[str] = None,
        session_id: Optional[str] = None,
        user_context: Optional[str] = None,
    ) -> ReviewItem:
        """
        Add a query to the review queue.

        Handles deduplication by merging similar questions.

        Args:
            question: The user's question
            reason: Why this needs review
            generated_sql: SQL that was generated (if any)
            semantic_evaluation: Evaluation result (if available)
            execution_result: Result from execution (if available)
            error_message: Any error that occurred
            session_id: Session identifier for tracking
            user_context: Additional context from user

        Returns:
            The created or updated ReviewItem
        """
        normalized_q = question.lower().strip()

        # Check for existing similar question
        existing_id = self._find_similar_question(normalized_q)

        if existing_id:
            # Update existing item
            item = self._items[existing_id]
            item.occurrence_count += 1
            item.updated_at = datetime.now().isoformat()

            # Keep the most recent SQL and evaluation
            if generated_sql:
                item.generated_sql = generated_sql
            if semantic_evaluation:
                item.semantic_evaluation = self._evaluation_to_dict(semantic_evaluation)
            if execution_result:
                item.execution_result = execution_result
            if error_message:
                item.error_message = error_message

            logger.info(
                f"Updated existing review item {existing_id} "
                f"(occurrences: {item.occurrence_count})"
            )
            return item

        # Create new item
        item_id = str(uuid4())[:8]
        item = ReviewItem(
            id=item_id,
            question=question,
            generated_sql=generated_sql,
            reason=reason,
            semantic_evaluation=self._evaluation_to_dict(semantic_evaluation)
            if semantic_evaluation else None,
            execution_result=execution_result,
            error_message=error_message,
            session_id=session_id,
            user_context=user_context,
        )

        self._items[item_id] = item
        self._question_index[normalized_q] = item_id

        # Enforce max queue size (remove oldest resolved items first)
        self._enforce_max_size()

        logger.info(f"Added new review item {item_id}: {reason.value}")
        return item

    def _evaluation_to_dict(self, evaluation: SemanticEvaluation) -> Dict[str, Any]:
        """Convert SemanticEvaluation to dict for storage."""
        return {
            "verdict": evaluation.verdict.value,
            "confidence": evaluation.confidence,
            "reasoning": evaluation.reasoning,
            "issues": evaluation.issues,
            "suggestions": evaluation.suggestions,
            "question_clarity_score": evaluation.question_clarity_score,
            "is_answerable": evaluation.is_answerable,
        }

    def _find_similar_question(self, normalized_question: str) -> Optional[str]:
        """Find existing item with similar question."""
        # Exact match first
        if normalized_question in self._question_index:
            return self._question_index[normalized_question]

        # Simple similarity check (could be enhanced with embeddings)
        for existing_q, item_id in self._question_index.items():
            if self._is_similar(normalized_question, existing_q):
                # Add to similar questions list
                item = self._items[item_id]
                if normalized_question not in item.similar_questions:
                    item.similar_questions.append(normalized_question)
                return item_id

        return None

    def _is_similar(self, q1: str, q2: str) -> bool:
        """Check if two questions are similar (simple implementation)."""
        # Simple word overlap similarity
        words1 = set(q1.split())
        words2 = set(q2.split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        similarity = intersection / union
        return similarity >= self.similarity_threshold

    def _enforce_max_size(self) -> None:
        """Remove old resolved items if queue exceeds max size."""
        if len(self._items) <= self.max_queue_size:
            return

        # Sort by status (resolved first) then by date
        sorted_items = sorted(
            self._items.values(),
            key=lambda x: (
                0 if x.status in [ReviewStatus.RESOLVED, ReviewStatus.REJECTED] else 1,
                x.created_at,
            ),
        )

        # Remove oldest resolved items
        while len(self._items) > self.max_queue_size:
            item = sorted_items.pop(0)
            if item.status in [ReviewStatus.RESOLVED, ReviewStatus.REJECTED]:
                del self._items[item.id]
                normalized_q = item.question.lower().strip()
                if normalized_q in self._question_index:
                    del self._question_index[normalized_q]
            else:
                break  # Don't remove pending items

    def get_item(self, item_id: str) -> Optional[ReviewItem]:
        """Get a review item by ID."""
        return self._items.get(item_id)

    def get_pending_items(self) -> List[ReviewItem]:
        """Get all pending review items, sorted by priority."""
        pending = [
            item for item in self._items.values()
            if item.status == ReviewStatus.PENDING
        ]

        # Sort by occurrence count (more frequent = higher priority)
        return sorted(pending, key=lambda x: -x.occurrence_count)

    def get_items_by_reason(self, reason: ReviewReason) -> List[ReviewItem]:
        """Get items filtered by reason."""
        return [
            item for item in self._items.values()
            if item.reason == reason
        ]

    def resolve_item(
        self,
        item_id: str,
        resolution_notes: str,
        resolved_sql: Optional[str] = None,
        resolved_by: Optional[str] = None,
        add_to_golden_set: bool = False,
    ) -> bool:
        """
        Mark an item as resolved.

        Args:
            item_id: ID of item to resolve
            resolution_notes: Notes about the resolution
            resolved_sql: Correct SQL for the question (if determined)
            resolved_by: Who resolved it
            add_to_golden_set: Whether to add to golden set

        Returns:
            True if item was found and resolved
        """
        item = self._items.get(item_id)
        if not item:
            return False

        item.status = (
            ReviewStatus.ADDED_TO_GOLDEN_SET
            if add_to_golden_set
            else ReviewStatus.RESOLVED
        )
        item.resolution_notes = resolution_notes
        item.resolved_sql = resolved_sql
        item.resolved_by = resolved_by
        item.resolved_at = datetime.now().isoformat()
        item.updated_at = datetime.now().isoformat()

        logger.info(f"Resolved review item {item_id}")
        return True

    def reject_item(
        self,
        item_id: str,
        reason: str,
        rejected_by: Optional[str] = None,
    ) -> bool:
        """
        Mark an item as rejected (question is intentionally unanswerable).

        Args:
            item_id: ID of item to reject
            reason: Why the question is being rejected
            rejected_by: Who rejected it

        Returns:
            True if item was found and rejected
        """
        item = self._items.get(item_id)
        if not item:
            return False

        item.status = ReviewStatus.REJECTED
        item.resolution_notes = f"REJECTED: {reason}"
        item.resolved_by = rejected_by
        item.resolved_at = datetime.now().isoformat()
        item.updated_at = datetime.now().isoformat()

        logger.info(f"Rejected review item {item_id}: {reason}")
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        items = list(self._items.values())

        status_counts = {status: 0 for status in ReviewStatus}
        reason_counts = {reason: 0 for reason in ReviewReason}

        for item in items:
            status_counts[item.status] += 1
            reason_counts[item.reason] += 1

        total_occurrences = sum(item.occurrence_count for item in items)
        pending_count = status_counts[ReviewStatus.PENDING]

        return {
            "total_items": len(items),
            "pending": pending_count,
            "resolved": status_counts[ReviewStatus.RESOLVED],
            "rejected": status_counts[ReviewStatus.REJECTED],
            "added_to_golden_set": status_counts[ReviewStatus.ADDED_TO_GOLDEN_SET],
            "total_occurrences": total_occurrences,
            "by_reason": {r.value: c for r, c in reason_counts.items()},
            "high_frequency_count": sum(
                1 for item in items if item.occurrence_count > 3
            ),
        }

    def export_for_review(self, output_path: Path) -> None:
        """
        Export pending items to a human-readable format for review.

        Args:
            output_path: Path to output file (markdown)
        """
        pending = self.get_pending_items()

        lines = [
            "# SQL Query Human Review Queue",
            "",
            f"Generated: {datetime.now().isoformat()}",
            f"Total pending items: {len(pending)}",
            "",
            "---",
            "",
        ]

        for item in pending:
            lines.extend([
                f"## Item: {item.id}",
                "",
                f"**Question:** {item.question}",
                f"**Reason:** {item.reason.value}",
                f"**Occurrences:** {item.occurrence_count}",
                f"**Created:** {item.created_at}",
                "",
            ])

            if item.generated_sql:
                lines.extend([
                    "**Generated SQL:**",
                    "```sql",
                    item.generated_sql,
                    "```",
                    "",
                ])

            if item.semantic_evaluation:
                eval_data = item.semantic_evaluation
                lines.extend([
                    "**Evaluation:**",
                    f"- Verdict: {eval_data.get('verdict', 'N/A')}",
                    f"- Confidence: {eval_data.get('confidence', 'N/A')}",
                    f"- Reasoning: {eval_data.get('reasoning', 'N/A')}",
                    "",
                ])

            if item.error_message:
                lines.append(f"**Error:** {item.error_message}")
                lines.append("")

            if item.similar_questions:
                lines.append("**Similar questions asked:**")
                for sq in item.similar_questions[:5]:
                    lines.append(f"- {sq}")
                lines.append("")

            lines.extend(["---", ""])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Exported {len(pending)} review items to {output_path}")


class AmbiguousQueryDetector:
    """
    Detects ambiguous queries that may need clarification.

    Uses heuristics and patterns to identify questions that are
    likely to be misinterpreted or cannot be answered clearly.
    """

    # Patterns indicating ambiguous queries
    VAGUE_PATTERNS = [
        "show me",
        "give me",
        "get the data",
        "list everything",
        "show all",
        "what about",
        "something with",
        "anything",
        "whatever",
        "the usual",
        "you know",
        "etc",
    ]

    # Questions that likely refer to non-database topics
    OFF_TOPIC_PATTERNS = [
        "weather",
        "stock",
        "news",
        "sports",
        "movie",
        "restaurant",
        "direction",
        "translate",
        "define",
        "meaning of",
        "how to cook",
        "recipe",
    ]

    # Minimum question length for validity
    MIN_QUESTION_LENGTH = 10

    def __init__(
        self,
        vague_patterns: Optional[List[str]] = None,
        off_topic_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize the detector.

        Args:
            vague_patterns: Custom vague patterns to detect
            off_topic_patterns: Custom off-topic patterns to detect
        """
        self.vague_patterns = vague_patterns or self.VAGUE_PATTERNS
        self.off_topic_patterns = off_topic_patterns or self.OFF_TOPIC_PATTERNS

    def analyze(self, question: str) -> Dict[str, Any]:
        """
        Analyze a question for ambiguity indicators.

        Args:
            question: The user's question

        Returns:
            Analysis result with detected issues
        """
        question_lower = question.lower().strip()

        issues = []
        confidence_reduction = 0.0

        # Check minimum length
        if len(question_lower) < self.MIN_QUESTION_LENGTH:
            issues.append("Question is too short to be specific")
            confidence_reduction += 0.3

        # Check for vague patterns
        for pattern in self.vague_patterns:
            if pattern in question_lower:
                issues.append(f"Vague phrase detected: '{pattern}'")
                confidence_reduction += 0.15

        # Check for off-topic patterns
        for pattern in self.off_topic_patterns:
            if pattern in question_lower:
                issues.append(f"Off-topic indicator: '{pattern}'")
                confidence_reduction += 0.4

        # Check for missing specifics (no nouns that could be table/column names)
        words = question_lower.split()
        specific_words = [
            w for w in words
            if len(w) > 3 and w not in [
                "what", "when", "where", "which", "show", "list",
                "give", "tell", "find", "with", "from", "that",
                "have", "does", "there", "many", "much", "some",
            ]
        ]
        if len(specific_words) < 2:
            issues.append("Question lacks specific terms (table/column names)")
            confidence_reduction += 0.2

        # Determine if question needs review
        needs_review = (
            len(issues) > 1
            or confidence_reduction > 0.3
            or any("off-topic" in issue.lower() for issue in issues)
        )

        return {
            "question": question,
            "is_ambiguous": len(issues) > 0,
            "needs_review": needs_review,
            "issues": issues,
            "confidence_adjustment": min(confidence_reduction, 0.5),
            "specific_words_found": specific_words,
        }

    def should_flag_for_review(
        self,
        question: str,
        semantic_evaluation: Optional[SemanticEvaluation] = None,
    ) -> tuple[bool, ReviewReason]:
        """
        Determine if a question should be flagged for human review.

        Args:
            question: The user's question
            semantic_evaluation: Optional semantic evaluation result

        Returns:
            Tuple of (should_flag, reason)
        """
        analysis = self.analyze(question)

        # Check semantic evaluation if available
        if semantic_evaluation:
            if semantic_evaluation.verdict == SemanticVerdict.AMBIGUOUS:
                return True, ReviewReason.AMBIGUOUS_QUESTION

            if semantic_evaluation.verdict == SemanticVerdict.INVALID_QUESTION:
                return True, ReviewReason.INVALID_QUESTION

            if semantic_evaluation.confidence < 0.6:
                return True, ReviewReason.LOW_CONFIDENCE

            if semantic_evaluation.question_clarity_score < 0.5:
                return True, ReviewReason.UNCLEAR_INTENT

        # Check heuristic analysis
        if analysis["needs_review"]:
            if any("off-topic" in issue.lower() for issue in analysis["issues"]):
                return True, ReviewReason.INVALID_QUESTION
            return True, ReviewReason.AMBIGUOUS_QUESTION

        return False, ReviewReason.AMBIGUOUS_QUESTION  # Default reason if flagged later
