"""Unit tests for PIIMasker."""

import pytest

from multi_tool_agent.tools.sql_accessor.pii_masker import (
    MaskingResult,
    PIIMasker,
)


class TestPIIMasker:
    """Tests for PIIMasker PII detection and masking."""

    @pytest.fixture
    def masker(self):
        """Create default PIIMasker."""
        return PIIMasker()

    @pytest.fixture
    def custom_masker(self):
        """Create PIIMasker with custom PII columns."""
        return PIIMasker(pii_columns={"email", "ssn", "credit_card"})

    # Email masking tests
    def test_mask_email_standard(self, masker):
        """Test masking of standard email address."""
        columns = ["id", "name", "email"]
        rows = [(1, "John Doe", "john.doe@example.com")]

        result = masker.mask_sql_result(columns, rows)

        assert result.rows[0][2] == "***@***.com"
        assert result.pii_stats.get("email") == 1

    def test_mask_email_different_tld(self, masker):
        """Test masking preserves TLD for different domains."""
        columns = ["email"]
        rows = [
            ("user@company.org",),
            ("test@domain.co.uk",),
        ]

        result = masker.mask_sql_result(columns, rows)

        assert result.rows[0][0] == "***@***.org"
        # Note: co.uk is treated as TLD
        assert "***@***" in result.rows[1][0]

    def test_mask_email_invalid_format(self, masker):
        """Test masking of email without @ symbol."""
        columns = ["email"]
        rows = [("not-an-email",)]

        result = masker.mask_sql_result(columns, rows)

        assert result.rows[0][0] == "***@***.***"

    # Phone masking tests
    def test_mask_phone_standard(self, masker):
        """Test masking of standard US phone number."""
        columns = ["id", "phone"]
        rows = [(1, "555-123-4567")]

        result = masker.mask_sql_result(columns, rows)

        assert result.rows[0][1] == "***-***-4567"
        assert result.pii_stats.get("phone") == 1

    def test_mask_phone_with_parentheses(self, masker):
        """Test masking of phone with area code in parentheses."""
        columns = ["phone"]
        rows = [("(555) 123-4567",)]

        result = masker.mask_sql_result(columns, rows)

        assert result.rows[0][0] == "***-***-4567"

    def test_mask_phone_digits_only(self, masker):
        """Test masking of phone with only digits."""
        columns = ["phone"]
        rows = [("5551234567",)]

        result = masker.mask_sql_result(columns, rows)

        assert result.rows[0][0] == "***-***-4567"

    def test_mask_phone_international(self, masker):
        """Test masking of international phone number."""
        columns = ["phone"]
        rows = [("+1-555-123-4567",)]

        result = masker.mask_sql_result(columns, rows)

        # Should keep last 4 digits
        assert "4567" in result.rows[0][0]

    def test_mask_phone_short(self, masker):
        """Test masking of short phone number."""
        columns = ["phone"]
        rows = [("123",)]

        result = masker.mask_sql_result(columns, rows)

        assert result.rows[0][0] == "***-***-****"

    # Address masking tests
    def test_mask_address(self, masker):
        """Test masking of street address."""
        columns = ["id", "address"]
        rows = [(1, "123 Main Street, Apt 4, New York, NY 10001")]

        result = masker.mask_sql_result(columns, rows)

        assert result.rows[0][1] == "[ADDRESS REDACTED]"
        assert result.pii_stats.get("address") == 1

    def test_mask_address_simple(self, masker):
        """Test masking of simple address."""
        columns = ["address"]
        rows = [("456 Oak Ave",)]

        result = masker.mask_sql_result(columns, rows)

        assert result.rows[0][0] == "[ADDRESS REDACTED]"

    # Multiple PII columns tests
    def test_mask_multiple_pii_columns(self, masker):
        """Test masking of multiple PII columns in same row."""
        columns = ["id", "name", "email", "phone", "address"]
        rows = [
            (1, "John Doe", "john@example.com", "555-123-4567", "123 Main St"),
        ]

        result = masker.mask_sql_result(columns, rows)

        assert result.rows[0][0] == 1  # id unchanged
        assert result.rows[0][1] == "John Doe"  # name unchanged
        assert result.rows[0][2] == "***@***.com"  # email masked
        assert result.rows[0][3] == "***-***-4567"  # phone masked
        assert result.rows[0][4] == "[ADDRESS REDACTED]"  # address masked

        assert result.pii_stats["email"] == 1
        assert result.pii_stats["phone"] == 1
        assert result.pii_stats["address"] == 1

    def test_mask_multiple_rows(self, masker):
        """Test masking across multiple rows."""
        columns = ["id", "email"]
        rows = [
            (1, "user1@example.com"),
            (2, "user2@example.com"),
            (3, "user3@example.com"),
        ]

        result = masker.mask_sql_result(columns, rows)

        assert len(result.rows) == 3
        assert all(row[1] == "***@***.com" for row in result.rows)
        assert result.pii_stats["email"] == 3

    # No PII columns tests
    def test_no_pii_columns(self, masker):
        """Test result when no PII columns present."""
        columns = ["id", "name", "category", "price"]
        rows = [
            (1, "Laptop", "Electronics", 999.99),
            (2, "Mouse", "Electronics", 29.99),
        ]

        result = masker.mask_sql_result(columns, rows)

        assert result.rows == rows  # Unchanged
        assert result.pii_stats == {}

    def test_empty_rows(self, masker):
        """Test handling of empty rows list."""
        columns = ["id", "email"]
        rows = []

        result = masker.mask_sql_result(columns, rows)

        assert result.rows == []
        assert result.pii_stats == {}

    # Null/None value tests
    def test_mask_null_values(self, masker):
        """Test handling of NULL values in PII columns."""
        columns = ["id", "email", "phone"]
        rows = [
            (1, None, "555-123-4567"),
            (2, "user@example.com", None),
        ]

        result = masker.mask_sql_result(columns, rows)

        assert result.rows[0][1] is None  # NULL unchanged
        assert result.rows[0][2] == "***-***-4567"
        assert result.rows[1][1] == "***@***.com"
        assert result.rows[1][2] is None  # NULL unchanged

    def test_mask_empty_string_values(self, masker):
        """Test handling of empty string values."""
        columns = ["email"]
        rows = [("",)]

        result = masker.mask_sql_result(columns, rows)

        assert result.rows[0][0] == ""  # Empty string unchanged

    def test_mask_none_string_values(self, masker):
        """Test handling of 'none' and 'null' string values."""
        columns = ["email"]
        rows = [
            ("none",),
            ("null",),
            ("None",),
        ]

        result = masker.mask_sql_result(columns, rows)

        # These should be left unchanged
        assert result.rows[0][0] == "none"
        assert result.rows[1][0] == "null"
        assert result.rows[2][0] == "None"

    # Case sensitivity tests
    def test_case_insensitive_column_names(self, masker):
        """Test that PII column detection is case-insensitive."""
        columns = ["ID", "EMAIL", "Phone", "ADDRESS"]
        rows = [(1, "user@example.com", "555-123-4567", "123 Main St")]

        result = masker.mask_sql_result(columns, rows)

        assert result.rows[0][1] == "***@***.com"
        assert result.rows[0][2] == "***-***-4567"
        assert result.rows[0][3] == "[ADDRESS REDACTED]"

    # Custom PII columns tests
    def test_custom_pii_columns(self, custom_masker):
        """Test masking with custom PII columns."""
        columns = ["id", "email", "ssn", "phone"]
        rows = [(1, "user@example.com", "123-45-6789", "555-123-4567")]

        result = custom_masker.mask_sql_result(columns, rows)

        assert result.rows[0][1] == "***@***.com"  # email masked
        assert result.rows[0][2] == "[REDACTED]"  # ssn masked (custom)
        assert result.rows[0][3] == "555-123-4567"  # phone NOT masked (not in custom set)

    # is_pii_column tests
    def test_is_pii_column_true(self, masker):
        """Test is_pii_column returns True for PII columns."""
        assert masker.is_pii_column("email") is True
        assert masker.is_pii_column("phone") is True
        assert masker.is_pii_column("address") is True
        assert masker.is_pii_column("EMAIL") is True  # Case insensitive

    def test_is_pii_column_false(self, masker):
        """Test is_pii_column returns False for non-PII columns."""
        assert masker.is_pii_column("id") is False
        assert masker.is_pii_column("name") is False
        assert masker.is_pii_column("price") is False
        assert masker.is_pii_column("email_verified") is False  # Partial match

    # Column index bounds tests
    def test_handles_mismatched_row_length(self, masker):
        """Test handling when row has fewer columns than expected."""
        columns = ["id", "email", "phone", "address"]
        rows = [(1, "user@example.com")]  # Missing phone and address

        result = masker.mask_sql_result(columns, rows)

        assert len(result.rows) == 1
        assert result.rows[0][1] == "***@***.com"


class TestMaskingResult:
    """Tests for MaskingResult dataclass."""

    def test_masking_result_creation(self):
        """Test MaskingResult creation."""
        result = MaskingResult(
            columns=["id", "email"],
            rows=[(1, "***@***.com")],
            pii_stats={"email": 1},
        )

        assert result.columns == ["id", "email"]
        assert len(result.rows) == 1
        assert result.pii_stats["email"] == 1

    def test_masking_result_empty_stats(self):
        """Test MaskingResult with empty stats."""
        result = MaskingResult(
            columns=["id", "name"],
            rows=[(1, "John")],
            pii_stats={},
        )

        assert result.pii_stats == {}
