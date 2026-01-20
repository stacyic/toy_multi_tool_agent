"""Unit tests for QueryChecker."""

import pytest

from multi_tool_agent.tools.sql_accessor.query_checker import (
    QueryChecker,
    ValidationResult,
)


class TestQueryChecker:
    """Tests for QueryChecker SQL validation."""

    @pytest.fixture
    def checker(self, sample_schema):
        """Create QueryChecker with sample schema."""
        return QueryChecker(sample_schema)

    # Valid query tests
    def test_valid_simple_select(self, checker):
        """Test validation of a simple SELECT query."""
        sql = "SELECT id, name FROM customers"
        result = checker.validate(sql)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert "customers" in result.tables_used

    def test_valid_select_with_where(self, checker):
        """Test validation of SELECT with WHERE clause."""
        sql = "SELECT name, email FROM customers WHERE id = 1"
        result = checker.validate(sql)

        assert result.is_valid is True
        assert "name" in result.columns_used
        assert "email" in result.columns_used

    def test_valid_select_with_join(self, checker):
        """Test validation of SELECT with JOIN."""
        sql = """
            SELECT c.name, o.total_amount
            FROM customers c
            JOIN orders o ON c.id = o.customer_id
        """
        result = checker.validate(sql)

        assert result.is_valid is True
        assert "customers" in result.tables_used
        assert "orders" in result.tables_used

    def test_valid_select_with_aggregation(self, checker):
        """Test validation of SELECT with aggregation."""
        sql = """
            SELECT customer_id, COUNT(*) as order_count, SUM(total_amount) as total
            FROM orders
            GROUP BY customer_id
        """
        result = checker.validate(sql)

        assert result.is_valid is True

    def test_valid_select_star(self, checker):
        """Test validation of SELECT * (should warn but be valid)."""
        sql = "SELECT * FROM products"
        result = checker.validate(sql)

        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("SELECT *" in w for w in result.warnings)

    def test_valid_select_with_limit(self, checker):
        """Test SELECT with LIMIT has no missing-LIMIT warning."""
        sql = "SELECT id, name FROM customers LIMIT 10"
        result = checker.validate(sql)

        assert result.is_valid is True
        # Should not have the "no LIMIT" warning
        limit_warnings = [w for w in result.warnings if "LIMIT" in w]
        assert len(limit_warnings) == 0

    def test_valid_subquery(self, checker):
        """Test validation of subquery."""
        sql = """
            SELECT name FROM customers
            WHERE id IN (SELECT customer_id FROM orders WHERE status = 'Delivered')
        """
        result = checker.validate(sql)

        assert result.is_valid is True

    # Invalid query tests - blocked operations
    def test_invalid_delete(self, checker):
        """Test that DELETE is blocked."""
        sql = "DELETE FROM customers WHERE id = 1"
        result = checker.validate(sql)

        assert result.is_valid is False
        assert any("Delete" in e for e in result.errors)

    def test_invalid_drop(self, checker):
        """Test that DROP is blocked."""
        sql = "DROP TABLE customers"
        result = checker.validate(sql)

        assert result.is_valid is False
        assert any("Drop" in e for e in result.errors)

    def test_invalid_insert(self, checker):
        """Test that INSERT is blocked."""
        sql = "INSERT INTO customers (name, email) VALUES ('Test', 'test@test.com')"
        result = checker.validate(sql)

        assert result.is_valid is False
        assert any("Insert" in e for e in result.errors)

    def test_invalid_update(self, checker):
        """Test that UPDATE is blocked."""
        sql = "UPDATE customers SET name = 'New Name' WHERE id = 1"
        result = checker.validate(sql)

        assert result.is_valid is False
        assert any("Update" in e for e in result.errors)

    def test_invalid_create(self, checker):
        """Test that CREATE is blocked."""
        sql = "CREATE TABLE new_table (id INTEGER)"
        result = checker.validate(sql)

        assert result.is_valid is False
        assert any("Create" in e for e in result.errors)

    def test_invalid_alter(self, checker):
        """Test that ALTER is blocked."""
        sql = "ALTER TABLE customers ADD COLUMN new_col TEXT"
        result = checker.validate(sql)

        assert result.is_valid is False
        assert any("Alter" in e for e in result.errors)

    # Invalid query tests - schema validation
    def test_invalid_unknown_table(self, checker):
        """Test detection of unknown table."""
        sql = "SELECT * FROM nonexistent_table"
        result = checker.validate(sql)

        assert result.is_valid is False
        assert any("nonexistent_table" in e.lower() for e in result.errors)

    def test_invalid_unknown_column(self, checker):
        """Test detection of unknown column."""
        sql = "SELECT nonexistent_column FROM customers"
        result = checker.validate(sql)

        assert result.is_valid is False
        assert any("nonexistent_column" in e.lower() for e in result.errors)

    def test_invalid_column_in_wrong_table(self, checker):
        """Test detection of column in wrong table."""
        sql = "SELECT customers.price FROM customers"
        result = checker.validate(sql)

        assert result.is_valid is False
        assert any("price" in e.lower() for e in result.errors)

    def test_invalid_syntax(self, checker):
        """Test detection of SQL syntax errors."""
        sql = "SELECT FROM WHERE"  # Invalid SQL structure
        result = checker.validate(sql)

        assert result.is_valid is False
        # Either syntax error or missing table/column errors
        assert len(result.errors) > 0

    # Suggestions tests
    def test_suggests_similar_table(self, checker):
        """Test that similar table names are suggested."""
        sql = "SELECT * FROM customer"  # Missing 's'
        result = checker.validate(sql)

        assert result.is_valid is False
        # Should suggest 'customers'
        errors_text = " ".join(result.errors).lower()
        assert "customers" in errors_text or "did you mean" in errors_text

    def test_suggests_similar_column(self, checker):
        """Test that similar column names are suggested."""
        sql = "SELECT nam FROM customers"  # Missing 'e'
        result = checker.validate(sql)

        assert result.is_valid is False
        # Should suggest 'name'
        errors_text = " ".join(result.errors).lower()
        assert "name" in errors_text or "did you mean" in errors_text

    # Error summary tests
    def test_get_error_summary_valid(self, checker):
        """Test error summary for valid query."""
        sql = "SELECT id FROM customers LIMIT 10"
        result = checker.validate(sql)
        summary = checker.get_error_summary(result)

        assert "valid" in summary.lower()

    def test_get_error_summary_invalid(self, checker):
        """Test error summary for invalid query."""
        sql = "SELECT * FROM nonexistent"
        result = checker.validate(sql)
        summary = checker.get_error_summary(result)

        assert "validation errors" in summary.lower()
        assert "nonexistent" in summary.lower()
        assert "valid tables" in summary.lower()

    # Edge cases
    def test_empty_query(self, checker):
        """Test handling of empty query."""
        sql = ""
        result = checker.validate(sql)

        assert result.is_valid is False

    def test_whitespace_only_query(self, checker):
        """Test handling of whitespace-only query."""
        sql = "   \n\t  "
        result = checker.validate(sql)

        assert result.is_valid is False

    def test_case_insensitive_table_names(self, checker):
        """Test that table names are case-insensitive."""
        sql = "SELECT id FROM CUSTOMERS"
        result = checker.validate(sql)

        assert result.is_valid is True

    def test_case_insensitive_column_names(self, checker):
        """Test that column names are case-insensitive."""
        sql = "SELECT ID, NAME FROM customers"
        result = checker.validate(sql)

        assert result.is_valid is True


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_values(self):
        """Test default values for ValidationResult."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.tables_used == set()
        assert result.columns_used == set()

    def test_with_errors(self):
        """Test ValidationResult with errors."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
            tables_used={"customers"},
        )

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert "customers" in result.tables_used
