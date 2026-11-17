"""A guarded, read-only SQL execution tool for the departments with real transactional data
(Billing, Shipping, Returns).

Generated SQL is parsed (not regex-matched) before it ever touches the database: it must be
a single SELECT statement that only references tables whitelisted for that department.
Anything else — stacked statements, DDL/DML, unlisted tables — raises SQLInjectionDetected
instead of running. That exception is a GuardrailViolation subclass, so it's automatically
caught by the shared guardrail wrapper in graph/build.py — no extra wiring needed here.
"""

import sqlite3

import sqlglot
from langchain_core.tools import tool
from sqlglot import expressions as exp

from shopunow.config import BUSINESS_DB_PATH
from shopunow.db.schema import ALLOWED_TABLES, DEPARTMENT_TABLES
from shopunow.guardrails.exceptions import SQLInjectionDetected

MAX_ROWS = 25


def validate_select(sql: str, allowed_tables: set[str]) -> None:
    """Raises SQLInjectionDetected if `sql` is anything other than a single SELECT
    statement that only touches `allowed_tables`."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except Exception as exc:
        raise SQLInjectionDetected(f"could not parse SQL: {exc}") from exc

    if len(statements) != 1:
        raise SQLInjectionDetected("only a single SELECT statement is allowed")

    stmt = statements[0]
    if not isinstance(stmt, exp.Select):
        raise SQLInjectionDetected(f"only SELECT statements are allowed, got {type(stmt).__name__}")

    referenced = {table.name.lower() for table in stmt.find_all(exp.Table)}
    unknown = referenced - {t.lower() for t in ALLOWED_TABLES}
    if unknown:
        raise SQLInjectionDetected(f"unknown table(s) referenced: {unknown}")
    not_permitted = referenced - {t.lower() for t in allowed_tables}
    if not_permitted:
        raise SQLInjectionDetected(f"table(s) not permitted for this department: {not_permitted}")


def execute_safe_select(sql: str, allowed_tables: set[str]) -> list[dict]:
    """Raises SQLInjectionDetected for a guard violation (handled upstream by the shared
    guardrail wrapper). A SQL statement that passes the guard but fails to execute — e.g. a
    typo'd column name — is a model mistake, not a security event, so that's reported back
    as a normal tool result instead, so the agent can respond gracefully."""
    validate_select(sql, allowed_tables)
    uri = f"file:{BUSINESS_DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql).fetchmany(MAX_ROWS)
        return [dict(row) for row in rows]
    except sqlite3.Error as exc:
        return [{"error": f"query failed to execute: {exc}"}]
    finally:
        conn.close()


def _schema_description(allowed_tables: set[str]) -> str:
    return "; ".join(
        f"{table}({', '.join(sorted(ALLOWED_TABLES[table]))})" for table in sorted(allowed_tables)
    )


def make_sql_tool(department_id: str):
    """Builds a @tool bound to one department's allowed table whitelist."""
    allowed_tables = DEPARTMENT_TABLES.get(department_id, set())

    def _query(sql: str) -> list[dict]:
        return execute_safe_select(sql, allowed_tables)

    _query.__name__ = f"query_{department_id}_database"
    _query.__doc__ = (
        f"Run a read-only SELECT query against the {department_id} department's tables. "
        f"Available tables and columns: {_schema_description(allowed_tables)}. Join on the "
        "matching id columns (e.g. orders.customer_id = customers.customer_id) to connect "
        "them. Always include a WHERE clause to look up a specific customer or order (e.g. "
        "by name, email, or order_id) rather than listing a whole table."
    )
    return tool(_query)
