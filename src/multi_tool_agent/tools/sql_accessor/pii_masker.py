"""PII masking for SQL query results.

This module provides column-name-based PII detection and masking.
It ONLY applies to database query results, not general text.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class MaskingResult:
    """Result of PII masking operation."""

    columns: List[str]
    rows: List[Tuple[Any, ...]]
    pii_stats: Dict[str, int]


class PIIMasker:
    """
    Column-name-based PII masking for SQL results.

    This class only masks data from SQL query results.
    External data (e.g., store addresses from web) should NOT be processed here.

    PII Detection Strategy:
    - Detects PII based on column names (email, phone, address)
    - Does not use regex/NLP on content (to avoid false positives)
    """

    # Column names that contain PII
    PII_COLUMNS: Set[str] = {"email", "phone", "address"}

    def __init__(self, pii_columns: Optional[Set[str]] = None):
        """
        Initialize the PII masker.

        Args:
            pii_columns: Optional custom set of PII column names.
                        Defaults to {"email", "phone", "address"}
        """
        self.pii_columns = pii_columns or self.PII_COLUMNS

    def mask_sql_result(
        self,
        columns: List[str],
        rows: List[Tuple[Any, ...]],
    ) -> MaskingResult:
        """
        Mask PII in SQL query results based on column names.

        Args:
            columns: Column names from the query result
            rows: Query result rows

        Returns:
            MaskingResult with masked rows and statistics
        """
        # Identify PII column indices
        pii_indices: Dict[int, str] = {}
        for i, col in enumerate(columns):
            col_lower = col.lower()
            if col_lower in self.pii_columns:
                pii_indices[i] = col_lower

        # No PII columns found
        if not pii_indices:
            return MaskingResult(
                columns=columns,
                rows=rows,
                pii_stats={},
            )

        # Mask PII values
        pii_stats: Dict[str, int] = {col: 0 for col in self.pii_columns}
        masked_rows: List[Tuple[Any, ...]] = []

        for row in rows:
            new_row = list(row)
            for idx, col_type in pii_indices.items():
                if idx < len(row) and row[idx] is not None:
                    new_row[idx] = self._mask_value(str(row[idx]), col_type)
                    pii_stats[col_type] += 1
            masked_rows.append(tuple(new_row))

        return MaskingResult(
            columns=columns,
            rows=masked_rows,
            pii_stats={k: v for k, v in pii_stats.items() if v > 0},
        )

    def _mask_value(self, value: str, pii_type: str) -> str:
        """
        Mask a PII value based on its type.

        Args:
            value: The value to mask
            pii_type: Type of PII (email, phone, address)

        Returns:
            Masked value
        """
        if not value or value.lower() in ("none", "null", ""):
            return value

        if pii_type == "email":
            return self._mask_email(value)
        elif pii_type == "phone":
            return self._mask_phone(value)
        elif pii_type == "address":
            return self._mask_address(value)
        else:
            return "[REDACTED]"

    def _mask_email(self, email: str) -> str:
        """
        Mask email address.

        Example: john.doe@example.com -> ***@***.com
        """
        if "@" not in email:
            return "***@***.***"

        parts = email.split("@")
        if len(parts) != 2:
            return "***@***.***"

        domain_parts = parts[1].rsplit(".", 1)
        if len(domain_parts) == 2:
            return f"***@***.{domain_parts[1]}"
        return "***@***.***"

    def _mask_phone(self, phone: str) -> str:
        """
        Mask phone number, keeping last 4 digits.

        Example: 555-123-4567 -> ***-***-4567
        """
        # Extract digits only
        digits = re.sub(r"\D", "", phone)

        if len(digits) >= 4:
            return f"***-***-{digits[-4:]}"
        return "***-***-****"

    def _mask_address(self, address: str) -> str:
        """
        Fully mask address.

        Example: 123 Main St, City, ST 12345 -> [ADDRESS REDACTED]
        """
        return "[ADDRESS REDACTED]"

    def is_pii_column(self, column_name: str) -> bool:
        """
        Check if a column name indicates PII.

        Args:
            column_name: Name of the column

        Returns:
            True if the column contains PII
        """
        return column_name.lower() in self.pii_columns
