"""Unit tests for SQLAccessor and related components."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from multi_tool_agent.tools.sql_accessor.sql_tool import (
    SQLAccessor,
    SQLAccessorResult,
)
from multi_tool_agent.tools.sql_accessor.query_executor import (
    ColumnInfo,
    ExecutionResult,
    QueryExecutor,
)
from multi_tool_agent.tools.sql_accessor.query_generator import (
    QueryGenerator,
    SQLGenerationResult,
)
from multi_tool_agent.core.exceptions import SQLRetryExhaustedError


class TestQueryExecutor:
    """Tests for QueryExecutor."""

    @pytest.fixture
    def executor(self, temp_db):
        """Create QueryExecutor with test database."""
        return QueryExecutor(str(temp_db))

    def test_introspect_schema(self, executor):
        """Test schema introspection."""
        schema = executor.introspect_schema()

        assert "customers" in schema
        assert "products" in schema
        assert "orders" in schema
        assert "order_items" in schema

        # Check customer columns
        customer_cols = {col.name for col in schema["customers"]}
        assert "id" in customer_cols
        assert "name" in customer_cols
        assert "email" in customer_cols

    def test_get_schema_description(self, executor):
        """Test schema description generation."""
        description = executor.get_schema_description()

        assert "customers" in description
        assert "products" in description
        assert "orders" in description

    @pytest.mark.asyncio
    async def test_execute_simple_query(self, executor):
        """Test executing a simple SELECT query."""
        result = await executor.execute("SELECT id, name FROM customers")

        assert result.success is True
        assert result.error is None
        assert len(result.columns) == 2
        assert "id" in result.columns
        assert "name" in result.columns
        assert len(result.rows) == 3  # 3 test customers

    @pytest.mark.asyncio
    async def test_execute_query_with_where(self, executor):
        """Test executing query with WHERE clause."""
        result = await executor.execute(
            "SELECT name FROM customers WHERE id = 1"
        )

        assert result.success is True
        assert len(result.rows) == 1
        assert result.rows[0][0] == "John Doe"

    @pytest.mark.asyncio
    async def test_execute_query_with_join(self, executor):
        """Test executing query with JOIN."""
        result = await executor.execute("""
            SELECT c.name, o.total_amount
            FROM customers c
            JOIN orders o ON c.id = o.customer_id
            ORDER BY o.id
        """)

        assert result.success is True
        assert len(result.rows) == 3  # 3 test orders

    @pytest.mark.asyncio
    async def test_execute_invalid_query(self, executor):
        """Test executing invalid query returns error."""
        result = await executor.execute("SELECT * FROM nonexistent_table")

        assert result.success is False
        assert result.error is not None
        assert "no such table" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_aggregation(self, executor):
        """Test executing aggregation query."""
        result = await executor.execute(
            "SELECT COUNT(*) as count FROM customers"
        )

        assert result.success is True
        assert result.rows[0][0] == 3


class TestQueryGenerator:
    """Tests for QueryGenerator."""

    @pytest.fixture
    def generator(self, mock_llm):
        """Create QueryGenerator with mocked LLM."""
        with patch("multi_tool_agent.tools.sql_accessor.query_generator.ChatOpenAI") as mock_class:
            mock_class.return_value = mock_llm
            gen = QueryGenerator(model="gpt-4", api_key="test-key")
            gen._llm = mock_llm
            gen.set_schema("customers: id, name, email\nproducts: id, name, price")
            return gen

    @pytest.mark.asyncio
    async def test_generate_sql(self, generator, mock_llm):
        """Test SQL generation from natural language."""
        result = await generator.generate("Show me all customers")

        assert result.sql is not None
        assert "SELECT" in result.sql

    @pytest.mark.asyncio
    async def test_generate_with_business_context(self, generator, mock_llm):
        """Test SQL generation with business context."""
        result = await generator.generate(
            "Show VIP customers",
            business_context="VIP customers have spent over $1000"
        )

        mock_llm.ainvoke.assert_called_once()
        call_args = mock_llm.ainvoke.call_args[0][0]
        # Check that context was included in messages
        assert any("VIP" in str(msg) for msg in call_args)

    @pytest.mark.asyncio
    async def test_generate_with_error_feedback(self, generator, mock_llm):
        """Test SQL generation with error feedback for retry."""
        result = await generator.generate(
            "Show customers",
            error_feedback="Column 'invalid' does not exist",
            previous_sql="SELECT invalid FROM customers"
        )

        mock_llm.ainvoke.assert_called_once()
        call_args = mock_llm.ainvoke.call_args[0][0]
        # Check that error feedback was included
        assert any("invalid" in str(msg).lower() for msg in call_args)

    def test_parse_json_response(self, generator):
        """Test parsing JSON response from LLM."""
        content = '{"sql": "SELECT * FROM test", "explanation": "Test query", "columns": ["id"]}'
        result = generator._parse_response(content)

        assert result.sql == "SELECT * FROM test"
        assert result.explanation == "Test query"
        assert result.requested_columns == ["id"]

    def test_parse_json_with_markdown(self, generator):
        """Test parsing JSON wrapped in markdown code blocks."""
        content = '```json\n{"sql": "SELECT * FROM test", "explanation": "Test", "columns": []}\n```'
        result = generator._parse_response(content)

        assert result.sql == "SELECT * FROM test"

    def test_parse_fallback_sql_extraction(self, generator):
        """Test fallback SQL extraction from non-JSON response."""
        content = "Here is the query: SELECT id, name FROM customers WHERE id = 1"
        result = generator._parse_response(content)

        assert "SELECT" in result.sql
        assert "customers" in result.sql

    @pytest.mark.asyncio
    async def test_schema_not_set_error(self):
        """Test error when schema not set."""
        with patch("multi_tool_agent.tools.sql_accessor.query_generator.ChatOpenAI"):
            gen = QueryGenerator(model="gpt-4", api_key="test")

        with pytest.raises(ValueError, match="Schema not set"):
            await gen.generate("test query")


class TestSQLAccessor:
    """Tests for SQLAccessor main tool."""

    @pytest.fixture
    def mock_executor(self):
        """Create mock QueryExecutor."""
        executor = MagicMock()
        executor.introspect_schema.return_value = {
            "customers": [
                ColumnInfo("id", "INTEGER", False, True),
                ColumnInfo("name", "TEXT", False, False),
                ColumnInfo("email", "TEXT", True, False),
            ]
        }
        executor.get_schema_description.return_value = "customers: id, name, email"
        executor.execute = AsyncMock(return_value=ExecutionResult(
            success=True,
            columns=["id", "name", "email"],
            rows=[(1, "John", "john@example.com")],
            row_count=1,
        ))
        return executor

    @pytest.fixture
    def mock_generator(self):
        """Create mock QueryGenerator."""
        generator = MagicMock()
        generator.generate = AsyncMock(return_value=SQLGenerationResult(
            sql="SELECT id, name, email FROM customers LIMIT 10",
            explanation="Query customers",
            requested_columns=["id", "name", "email"],
        ))
        generator.set_schema = MagicMock()
        return generator

    @pytest.fixture
    def sql_accessor(self, temp_db, mock_executor, mock_generator):
        """Create SQLAccessor with mocked components."""
        with patch("multi_tool_agent.tools.sql_accessor.sql_tool.QueryExecutor") as mock_exec_class, \
             patch("multi_tool_agent.tools.sql_accessor.sql_tool.QueryGenerator") as mock_gen_class:
            mock_exec_class.return_value = mock_executor
            mock_gen_class.return_value = mock_generator

            accessor = SQLAccessor(
                db_path=str(temp_db),
                model="gpt-4",
                api_key="test-key",
                max_retries=3,
            )
            accessor.executor = mock_executor
            accessor.generator = mock_generator
            return accessor

    @pytest.mark.asyncio
    async def test_run_successful_query(self, sql_accessor):
        """Test successful query execution."""
        result = await sql_accessor.run("Show me all customers")

        assert result.success is True
        assert result.data is not None
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_run_masks_pii(self, sql_accessor, mock_executor):
        """Test that PII is masked in results."""
        mock_executor.execute = AsyncMock(return_value=ExecutionResult(
            success=True,
            columns=["id", "name", "email"],
            rows=[(1, "John", "john@example.com")],
            row_count=1,
        ))

        result = await sql_accessor.run("Show customer emails")

        assert result.success is True
        assert "***@***.com" in result.data  # Email should be masked
        assert result.pii_masked is not None
        assert result.pii_masked.get("email", 0) > 0

    @pytest.mark.asyncio
    async def test_run_with_validation_retry(self, sql_accessor, mock_generator):
        """Test that validation errors trigger regeneration (free retry)."""
        # First generation returns invalid SQL, second is valid
        mock_generator.generate = AsyncMock(side_effect=[
            SQLGenerationResult(
                sql="SELECT invalid_column FROM customers",
                explanation="",
                requested_columns=[],
            ),
            SQLGenerationResult(
                sql="SELECT id, name FROM customers LIMIT 10",
                explanation="",
                requested_columns=["id", "name"],
            ),
        ])

        result = await sql_accessor.run("Show customers")

        # Should succeed on second try (validation retry is free)
        assert result.success is True
        assert mock_generator.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_run_with_execution_retry(self, sql_accessor, mock_generator, mock_executor):
        """Test that execution errors trigger retry."""
        # First execution fails, second succeeds
        mock_executor.execute = AsyncMock(side_effect=[
            ExecutionResult(
                success=False,
                error="SQLite error: something went wrong",
            ),
            ExecutionResult(
                success=True,
                columns=["id"],
                rows=[(1,)],
                row_count=1,
            ),
        ])

        result = await sql_accessor.run("Show customers")

        assert result.success is True
        assert result.attempts == 2  # Two execution attempts

    @pytest.mark.asyncio
    async def test_run_exhausts_retries(self, sql_accessor, mock_executor):
        """Test that exhausted retries return failed result."""
        mock_executor.execute = AsyncMock(return_value=ExecutionResult(
            success=False,
            error="Persistent error",
        ))

        result = await sql_accessor.run("Show customers")

        assert result.success is False
        assert result.attempts == 3  # max_retries
        assert "Persistent error" in result.error

    @pytest.mark.asyncio
    async def test_run_with_context(self, sql_accessor, mock_generator):
        """Test that context is passed to generator."""
        context = {
            "policy_accessor": "VIP = customers with >$1000 spent"
        }

        await sql_accessor.run("How many VIP customers?", context=context)

        # Check that context was passed to generate
        call_kwargs = mock_generator.generate.call_args[1]
        assert call_kwargs.get("business_context") == "VIP = customers with >$1000 spent"

    @pytest.mark.asyncio
    async def test_run_generation_failure(self, sql_accessor, mock_generator):
        """Test handling of generation failure."""
        mock_generator.generate = AsyncMock(
            side_effect=Exception("API error")
        )

        result = await sql_accessor.run("Show customers")

        assert result.success is False
        assert "generation failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_run_empty_sql(self, sql_accessor, mock_generator):
        """Test handling of empty SQL generation returns apologetic error."""
        mock_generator.generate = AsyncMock(return_value=SQLGenerationResult(
            sql="",
            explanation="",
            requested_columns=[],
        ))

        result = await sql_accessor.run("Show customers")

        assert result.success is False
        assert "unable to generate a valid database query" in result.error
        assert "3 attempts" in result.error


class TestSQLAccessorResult:
    """Tests for SQLAccessorResult dataclass."""

    def test_success_result(self):
        """Test creating a success result."""
        result = SQLAccessorResult(
            success=True,
            data="Results (1 rows):\nid | name\n---\n1 | John",
            sql_executed="SELECT id, name FROM customers",
            attempts=1,
        )

        assert result.success is True
        assert result.error is None
        assert "John" in result.data

    def test_error_result(self):
        """Test creating an error result."""
        result = SQLAccessorResult(
            success=False,
            error="Query failed",
            attempts=3,
        )

        assert result.success is False
        assert result.data is None
        assert result.error == "Query failed"

    def test_result_with_pii_stats(self):
        """Test result with PII masking stats."""
        result = SQLAccessorResult(
            success=True,
            data="Results...",
            pii_masked={"email": 5, "phone": 3},
            attempts=1,
        )

        assert result.pii_masked["email"] == 5
        assert result.pii_masked["phone"] == 3


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_success_execution(self):
        """Test successful execution result."""
        result = ExecutionResult(
            success=True,
            columns=["id", "name"],
            rows=[(1, "John"), (2, "Jane")],
            row_count=2,
        )

        assert result.success is True
        assert len(result.rows) == 2

    def test_failed_execution(self):
        """Test failed execution result."""
        result = ExecutionResult(
            success=False,
            error="no such table: invalid",
        )

        assert result.success is False
        assert "no such table" in result.error


class TestColumnInfo:
    """Tests for ColumnInfo dataclass."""

    def test_column_info(self):
        """Test ColumnInfo creation."""
        col = ColumnInfo(name="id", type="INTEGER", nullable=False, primary_key=True)

        assert col.name == "id"
        assert col.type == "INTEGER"
        assert col.nullable is False
        assert col.primary_key is True
