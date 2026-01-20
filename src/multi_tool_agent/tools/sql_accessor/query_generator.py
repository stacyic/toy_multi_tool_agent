"""SQL query generator using LLM."""

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from langchain_openai import ChatOpenAI

from ...utils.llm_utils import invoke_with_logging

if TYPE_CHECKING:
    from ...logging.trace_logger import TraceLogger


@dataclass
class SQLGenerationResult:
    """Result from SQL generation."""

    sql: str
    explanation: str
    requested_columns: List[str]


class QueryGenerator:
    """
    Generates SQL from natural language using LLM.

    Features:
    - Schema-aware generation
    - Business context injection for domain-specific terms
    - Self-correction with error feedback
    """

    SYSTEM_PROMPT = """You are a SQL expert. Generate SQLite-compatible SQL queries based on natural language questions.

DATABASE SCHEMA:
{schema}

{business_context}

RULES:
1. Only use tables and columns that exist in the schema above
2. Use proper SQLite syntax and functions
3. For date comparisons, use SQLite date functions like date(), datetime(), strftime()
4. Always limit results to a reasonable amount (default 100 rows) unless counting
5. Use JOINs appropriately when querying related tables
6. For aggregate queries, use GROUP BY with appropriate columns
7. Column names are case-insensitive in SQLite
8. For name searches, use LIKE with wildcards for partial matching (e.g., WHERE name LIKE '%Alex%') since users often provide partial names
9. When a query could match multiple records (e.g., searching by first name), return ALL matching records rather than assuming a single match
10. CRITICAL - SQLite type consistency: strftime() returns TEXT, so when comparing dates:
    - CORRECT: strftime('%Y', order_date) = strftime('%Y', 'now', '-1 year')  -- both TEXT
    - CORRECT: strftime('%Y', order_date) = '2025'  -- both TEXT
    - WRONG: strftime('%Y', order_date) = CAST(strftime('%Y', 'now') AS INTEGER) - 1  -- TEXT vs INTEGER fails!
    - For "previous/last calendar year", use: strftime('%Y', date_column) = strftime('%Y', 'now', '-1 year')

OUTPUT FORMAT (JSON):
{{
    "sql": "YOUR SQL QUERY HERE",
    "explanation": "Brief explanation of what the query does",
    "columns": ["list", "of", "columns", "in", "result"]
}}

Only output valid JSON. Do not include any other text or markdown formatting."""

    ERROR_CORRECTION_PROMPT = """The previous SQL query failed. Please correct it.

Previous Query:
{previous_sql}

Error:
{error}

Generate a corrected query following the same JSON format."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        logger: Optional["TraceLogger"] = None,
    ):
        """
        Initialize the query generator.

        Args:
            model: OpenAI model name
            api_key: OpenAI API key (optional)
            temperature: LLM temperature (0 for deterministic)
            logger: Optional TraceLogger for API call logging
        """
        self.model = model
        self.logger = logger
        kwargs = {"model": model, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        self._llm = ChatOpenAI(**kwargs)
        self._schema_description: Optional[str] = None

    def set_schema(self, schema_description: str) -> None:
        """
        Set the database schema description.

        Args:
            schema_description: Human-readable schema description
        """
        self._schema_description = schema_description

    async def generate(
        self,
        question: str,
        business_context: Optional[str] = None,
        error_feedback: Optional[str] = None,
        previous_sql: Optional[str] = None,
    ) -> SQLGenerationResult:
        """
        Generate SQL from natural language.

        Args:
            question: Natural language query
            business_context: Optional business context from PolicyAccessor
            error_feedback: Previous error message for self-correction
            previous_sql: Previous SQL that failed

        Returns:
            SQLGenerationResult with SQL, explanation, and expected columns
        """
        if not self._schema_description:
            raise ValueError("Schema not set. Call set_schema() first.")

        # Build business context section
        context_section = ""
        if business_context:
            context_section = f"""
BUSINESS CONTEXT:
The following business rules and definitions apply to this query:
{business_context}

Use this context to understand domain-specific terms in the question.
"""

        # Build the system prompt
        system_prompt = self.SYSTEM_PROMPT.format(
            schema=self._schema_description,
            business_context=context_section,
        )

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add error correction context if retrying
        if error_feedback and previous_sql:
            correction_msg = self.ERROR_CORRECTION_PROMPT.format(
                previous_sql=previous_sql,
                error=error_feedback,
            )
            messages.append({"role": "user", "content": correction_msg})
        else:
            messages.append({"role": "user", "content": f"Question: {question}"})

        # Generate response with logging
        response = await invoke_with_logging(
            llm=self._llm,
            messages=messages,
            logger=self.logger,
            component="sql_generation",
            model=self.model,
        )
        content = response.content

        # Parse JSON response
        return self._parse_response(content)

    def _parse_response(self, content: str) -> SQLGenerationResult:
        """
        Parse LLM response to extract SQL and metadata.

        Args:
            content: Raw LLM response

        Returns:
            SQLGenerationResult
        """
        # Try to extract JSON from response
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            # Find the JSON content
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
            if match:
                content = match.group(1).strip()

        try:
            data = json.loads(content)
            return SQLGenerationResult(
                sql=data.get("sql", "").strip(),
                explanation=data.get("explanation", ""),
                requested_columns=data.get("columns", []),
            )
        except json.JSONDecodeError:
            # Fallback: try to extract SQL directly
            sql_match = re.search(r"SELECT[\s\S]+?(?:;|$)", content, re.IGNORECASE)
            if sql_match:
                return SQLGenerationResult(
                    sql=sql_match.group(0).strip().rstrip(";") + "",
                    explanation="Extracted from non-JSON response",
                    requested_columns=[],
                )

            raise ValueError(f"Could not parse SQL from response: {content[:200]}")
