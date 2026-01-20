"""SQL Accessor tool with Text-to-SQL, self-correction, and semantic validation."""

from .sql_tool import SQLAccessor, SQLAccessorResult
from .query_generator import QueryGenerator
from .query_checker import QueryChecker
from .query_executor import QueryExecutor
from .pii_masker import PIIMasker
from .semantic_evaluator import SemanticEvaluator, SemanticEvaluation, SemanticVerdict
from .golden_set import GoldenSetManager, GoldenSetEntry, AccuracyMetrics, EvaluationResult
from .human_review import HumanReviewQueue, AmbiguousQueryDetector, ReviewItem, ReviewReason
from .accuracy_tracker import AccuracyTracker

__all__ = [
    # Core SQL tools
    "SQLAccessor",
    "SQLAccessorResult",
    "QueryGenerator",
    "QueryChecker",
    "QueryExecutor",
    "PIIMasker",
    # Semantic evaluation
    "SemanticEvaluator",
    "SemanticEvaluation",
    "SemanticVerdict",
    # Golden set evaluation
    "GoldenSetManager",
    "GoldenSetEntry",
    "AccuracyMetrics",
    "EvaluationResult",
    "AccuracyTracker",
    # Human review
    "HumanReviewQueue",
    "AmbiguousQueryDetector",
    "ReviewItem",
    "ReviewReason",
]
