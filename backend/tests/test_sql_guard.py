import pytest

from shopunow.guardrails.exceptions import SQLInjectionDetected
from shopunow.tools import sql_tool

RETURNS_TABLES = {"customers", "orders", "order_items", "inventory"}


def test_legit_select_passes():
    sql_tool.validate_select("SELECT * FROM orders WHERE order_id = 1001", RETURNS_TABLES)


def test_legit_join_passes():
    sql_tool.validate_select(
        "SELECT o.order_id, c.name FROM orders o JOIN customers c ON o.customer_id = c.customer_id",
        RETURNS_TABLES,
    )


@pytest.mark.parametrize(
    "malicious_sql",
    [
        "SELECT * FROM orders; DROP TABLE orders;--",
        "DROP TABLE orders",
        "DELETE FROM orders WHERE 1=1",
        "UPDATE orders SET status = 'cancelled'",
        "INSERT INTO orders (order_id) VALUES (9999)",
        "SELECT * FROM orders; SELECT * FROM customers",
        "PRAGMA table_info(orders)",
    ],
)
def test_malicious_sql_blocked(malicious_sql):
    with pytest.raises(SQLInjectionDetected):
        sql_tool.validate_select(malicious_sql, RETURNS_TABLES)


def test_table_not_in_department_whitelist_blocked():
    # payments is a real table, but not whitelisted for the returns department
    with pytest.raises(SQLInjectionDetected):
        sql_tool.validate_select("SELECT * FROM payments", RETURNS_TABLES)


def test_unknown_table_blocked():
    with pytest.raises(SQLInjectionDetected):
        sql_tool.validate_select("SELECT * FROM sqlite_master", RETURNS_TABLES)


def test_execute_safe_select_runs_against_real_db(seeded_sqlite_db, monkeypatch):
    monkeypatch.setattr(sql_tool, "BUSINESS_DB_PATH", seeded_sqlite_db)
    rows = sql_tool.execute_safe_select(
        "SELECT order_id, status FROM orders WHERE order_id = 1001", RETURNS_TABLES
    )
    assert rows == [{"order_id": 1001, "status": "delivered"}]


def test_execute_safe_select_rejects_injection_before_touching_db(seeded_sqlite_db, monkeypatch):
    monkeypatch.setattr(sql_tool, "BUSINESS_DB_PATH", seeded_sqlite_db)
    with pytest.raises(SQLInjectionDetected):
        sql_tool.execute_safe_select("SELECT * FROM orders; DROP TABLE orders;--", RETURNS_TABLES)

    # confirm the table really is untouched
    rows = sql_tool.execute_safe_select("SELECT order_id FROM orders WHERE order_id = 1001", RETURNS_TABLES)
    assert len(rows) == 1


def test_execute_safe_select_reports_execution_errors_gracefully_not_as_a_crash(seeded_sqlite_db, monkeypatch):
    monkeypatch.setattr(sql_tool, "BUSINESS_DB_PATH", seeded_sqlite_db)
    # a column that doesn't exist passes the guard (it's a real table, valid SELECT) but
    # fails at execution time — this must come back as a normal result, not raise
    rows = sql_tool.execute_safe_select("SELECT no_such_column FROM orders", RETURNS_TABLES)
    assert "error" in rows[0]
