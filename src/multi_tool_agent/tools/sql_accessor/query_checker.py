"""AST-based SQL validation using sqlglot."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import sqlglot
from sqlglot import exp


@dataclass
class ValidationResult:
    """Result of SQL validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tables_used: Set[str] = field(default_factory=set)
    columns_used: Set[str] = field(default_factory=set)


class QueryChecker:
    """
    AST-based SQL validation using sqlglot.

    Features:
    - Validates table/column existence before execution
    - Blocks dangerous operations (DELETE, DROP, etc.)
    - Provides detailed error messages for self-correction
    """

    # Operations that are not allowed
    BLOCKED_OPERATIONS = {
        exp.Drop,
        exp.Delete,
        exp.Insert,
        exp.Update,
        exp.Create,
        exp.Alter,
    }

    def __init__(self, schema_info: Dict[str, List[Dict]]):
        """
        Initialize the query checker.

        Args:
            schema_info: Schema dictionary mapping table names to column info.
                        Each column info dict should have 'name' key.
        """
        self.schema_info = schema_info
        self.valid_tables = {t.lower() for t in schema_info.keys()}
        self.valid_columns = self._build_column_map()
        self.all_columns = self._build_all_columns()

    def _build_column_map(self) -> Dict[str, Set[str]]:
        """Build table -> columns mapping."""
        return {
            table.lower(): {col["name"].lower() for col in columns}
            for table, columns in self.schema_info.items()
        }

    def _build_all_columns(self) -> Set[str]:
        """Build set of all valid column names."""
        all_cols = set()
        for columns in self.valid_columns.values():
            all_cols.update(columns)
        return all_cols

    def validate(self, sql: str) -> ValidationResult:
        """
        Validate SQL query against schema.

        This is a "free" pre-check that doesn't count toward retry limits.

        Args:
            sql: SQL query to validate

        Returns:
            ValidationResult with validity status and any errors
        """
        errors: List[str] = []
        warnings: List[str] = []
        tables_used: Set[str] = set()
        columns_used: Set[str] = set()

        try:
            # Parse SQL into AST
            parsed = sqlglot.parse_one(sql, dialect="sqlite")

            # Check for blocked operations
            for blocked_type in self.BLOCKED_OPERATIONS:
                if parsed.find(blocked_type):
                    errors.append(
                        f"{blocked_type.__name__} operations are not allowed"
                    )

            # Extract and validate table references
            for table in parsed.find_all(exp.Table):
                table_name = table.name.lower()
                tables_used.add(table_name)

                if table_name not in self.valid_tables:
                    errors.append(f"Unknown table: '{table_name}'")
                    # Suggest similar tables
                    suggestions = self._find_similar(table_name, self.valid_tables)
                    if suggestions:
                        errors[-1] += f". Did you mean: {', '.join(suggestions)}?"

            # Extract and validate column references
            for column in parsed.find_all(exp.Column):
                col_name = column.name.lower()
                table_ref = column.table.lower() if column.table else None
                columns_used.add(col_name)

                # Skip * (SELECT *)
                if col_name == "*":
                    continue

                if table_ref:
                    # Column with table prefix
                    if table_ref in self.valid_columns:
                        if col_name not in self.valid_columns[table_ref]:
                            errors.append(
                                f"Column '{col_name}' not found in table '{table_ref}'"
                            )
                            # Suggest similar columns
                            suggestions = self._find_similar(
                                col_name, self.valid_columns[table_ref]
                            )
                            if suggestions:
                                errors[-1] += f". Did you mean: {', '.join(suggestions)}?"
                    elif table_ref in tables_used:
                        # Table alias - we can't fully validate without alias tracking
                        # Just check if column exists anywhere
                        if col_name not in self.all_columns:
                            errors.append(f"Unknown column: '{col_name}'")
                else:
                    # Column without table prefix
                    if col_name not in self.all_columns:
                        errors.append(f"Unknown column: '{col_name}'")
                        suggestions = self._find_similar(col_name, self.all_columns)
                        if suggestions:
                            errors[-1] += f". Did you mean: {', '.join(suggestions)}?"

            # Check for common issues
            self._check_common_issues(parsed, warnings)

        except sqlglot.errors.ParseError as e:
            errors.append(f"SQL syntax error: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            tables_used=tables_used,
            columns_used=columns_used,
        )

    def _check_common_issues(
        self, parsed: exp.Expression, warnings: List[str]
    ) -> None:
        """Check for common SQL issues and add warnings."""
        # Check for SELECT * (might expose PII)
        for select in parsed.find_all(exp.Select):
            for col in select.expressions:
                if isinstance(col, exp.Star):
                    warnings.append(
                        "SELECT * may expose sensitive columns. "
                        "Consider selecting specific columns."
                    )
                    break

        # Check for missing LIMIT on SELECT
        if isinstance(parsed, exp.Select) and not parsed.find(exp.Limit):
            warnings.append(
                "Query has no LIMIT clause. Consider adding LIMIT to prevent "
                "returning too many rows."
            )

    def _find_similar(
        self, name: str, valid_names: Set[str], max_suggestions: int = 3
    ) -> List[str]:
        """Find similar names for suggestions."""
        suggestions = []
        name_lower = name.lower()

        for valid in valid_names:
            # Check for substring match
            if name_lower in valid or valid in name_lower:
                suggestions.append(valid)
            # Check for common typos (simple edit distance approximation)
            elif self._is_similar(name_lower, valid):
                suggestions.append(valid)

            if len(suggestions) >= max_suggestions:
                break

        return suggestions

    def _is_similar(self, s1: str, s2: str, threshold: float = 0.6) -> bool:
        """Check if two strings are similar using simple heuristics."""
        if not s1 or not s2:
            return False

        # Same length and differ by at most 2 characters
        if abs(len(s1) - len(s2)) <= 2:
            matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))
            return matches / max(len(s1), len(s2)) >= threshold

        return False

    def get_error_summary(self, result: ValidationResult) -> str:
        """
        Get a summary of validation errors suitable for LLM feedback.

        Args:
            result: ValidationResult from validate()

        Returns:
            Error summary string
        """
        if result.is_valid:
            return "Query is valid."

        lines = ["Validation errors:"]
        for error in result.errors:
            lines.append(f"- {error}")

        if result.warnings:
            lines.append("\nWarnings:")
            for warning in result.warnings:
                lines.append(f"- {warning}")

        lines.append(f"\nValid tables: {', '.join(sorted(self.valid_tables))}")

        return "\n".join(lines)
