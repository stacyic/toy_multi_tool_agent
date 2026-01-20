"""SQL query executor with schema introspection."""

import asyncio
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ColumnInfo:
    """Information about a database column."""

    name: str
    type: str
    nullable: bool
    primary_key: bool
    default: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result from SQL execution."""

    success: bool
    columns: List[str] = field(default_factory=list)
    rows: List[Tuple[Any, ...]] = field(default_factory=list)
    row_count: int = 0
    error: Optional[str] = None
    sql: Optional[str] = None


class QueryExecutor:
    """
    Executes validated SQL queries against SQLite.

    Features:
    - Schema introspection at startup (DB is source of truth)
    - Async wrapper around sync SQLite
    - Connection management
    """

    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize the query executor.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._schema: Optional[Dict[str, List[ColumnInfo]]] = None

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def introspect_schema(self) -> Dict[str, List[ColumnInfo]]:
        """
        Extract schema from SQLite database.

        The database is the source of truth for schema information.

        Returns:
            Dictionary mapping table names to list of ColumnInfo
        """
        if self._schema is not None:
            return self._schema

        schema: Dict[str, List[ColumnInfo]] = {}

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get all tables
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = []
                for row in cursor.fetchall():
                    columns.append(
                        ColumnInfo(
                            name=row[1],
                            type=row[2] or "TEXT",
                            nullable=not bool(row[3]),
                            primary_key=bool(row[5]),
                            default=row[4],
                        )
                    )
                schema[table] = columns

        self._schema = schema
        return schema

    def get_schema_description(self) -> str:
        """
        Get a human-readable schema description for LLM prompts.

        Returns:
            Formatted string describing the database schema
        """
        schema = self.introspect_schema()
        lines = []

        for table, columns in schema.items():
            col_descriptions = []
            for col in columns:
                desc = f"{col.name} ({col.type})"
                if col.primary_key:
                    desc += " PRIMARY KEY"
                if not col.nullable:
                    desc += " NOT NULL"
                col_descriptions.append(desc)

            lines.append(f"- {table}: {', '.join(col_descriptions)}")

        return "\n".join(lines)

    def get_valid_tables(self) -> set[str]:
        """Get set of valid table names."""
        return set(self.introspect_schema().keys())

    def get_valid_columns(self, table: Optional[str] = None) -> set[str]:
        """
        Get set of valid column names.

        Args:
            table: Optional table name to filter columns

        Returns:
            Set of column names
        """
        schema = self.introspect_schema()

        if table:
            if table not in schema:
                return set()
            return {col.name for col in schema[table]}

        # All columns from all tables
        all_columns = set()
        for columns in schema.values():
            all_columns.update(col.name for col in columns)
        return all_columns

    async def execute(self, sql: str) -> ExecutionResult:
        """
        Execute SQL query asynchronously.

        Uses run_in_executor for non-blocking IO.

        Args:
            sql: SQL query to execute

        Returns:
            ExecutionResult with success status, columns, rows, or error
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_sync, sql)

    def _execute_sync(self, sql: str) -> ExecutionResult:
        """
        Synchronous execution implementation.

        Args:
            sql: SQL query to execute

        Returns:
            ExecutionResult
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(sql)

                # Get column names from cursor description
                columns = (
                    [desc[0] for desc in cursor.description]
                    if cursor.description
                    else []
                )

                # Fetch all rows
                rows = [tuple(row) for row in cursor.fetchall()]

                return ExecutionResult(
                    success=True,
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    sql=sql,
                )

        except sqlite3.Error as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                sql=sql,
            )

    def execute_sync(self, sql: str) -> ExecutionResult:
        """
        Public synchronous execution method.

        Args:
            sql: SQL query to execute

        Returns:
            ExecutionResult
        """
        return self._execute_sync(sql)
