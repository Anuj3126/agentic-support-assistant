import sqlite3

import pytest

from shopunow.db.schema import SCHEMA_SQL


@pytest.fixture
def seeded_sqlite_db(tmp_path):
    """A tiny but real SQLite db with the project schema plus a handful of rows, used to
    test the SQL guard, identity resolution, and the returns write action against a real
    (temporary, isolated) database — never the shared dev shopunow.sqlite."""
    db_path = tmp_path / "test_shopunow.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.execute("INSERT INTO customers (customer_id, name, email) VALUES (1, 'Priya Sharma', 'priya@example.com')")
    conn.execute("INSERT INTO customers (customer_id, name, email) VALUES (2, 'Dev Gupta', 'dev@example.com')")
    conn.execute(
        "INSERT INTO inventory (item_id, name, category, price, stock_qty) "
        "VALUES (1, 'Classic Denim Jacket', 'clothing', 2499.0, 20)"
    )
    conn.execute(
        "INSERT INTO orders (order_id, customer_id, order_date, status) VALUES (1001, 1, '2026-01-06', 'delivered')"
    )
    conn.execute(
        "INSERT INTO orders (order_id, customer_id, order_date, status) VALUES (1003, 1, '2026-01-10', 'shipped')"
    )
    conn.execute(
        "INSERT INTO order_items (order_item_id, order_id, item_id, quantity, status) VALUES (1, 1001, 1, 1, 'active')"
    )
    conn.execute(
        "INSERT INTO order_items (order_item_id, order_id, item_id, quantity, status) VALUES (2, 1003, 1, 1, 'active')"
    )
    conn.execute("INSERT INTO employees (employee_id, name, email, department) VALUES (1, 'Asha Nair', 'asha@shopunow.com', 'hr')")
    conn.execute("INSERT INTO leave_balances (leave_balance_id, employee_id, leave_type, balance_days) VALUES (1, 1, 'PTO', 1.5)")
    conn.execute(
        "INSERT INTO it_tickets (ticket_id, employee_id, issue_type, status, created_date) "
        "VALUES (1, 1, 'vpn_access', 'open', '2026-01-10')"
    )
    conn.commit()
    conn.close()
    return db_path


class FakeStructuredLLM:
    """Stands in for `llm.with_structured_output(Model).invoke(prompt)` in unit tests,
    so agent node tests don't need a real GOOGLE_API_KEY or network call."""

    def __init__(self, response):
        self._response = response

    def invoke(self, _prompt):
        return self._response


class FakeChatModel:
    """Stands in for ChatGoogleGenerativeAI(...) — returns a FakeStructuredLLM regardless
    of which Pydantic schema is requested, since tests control the canned response directly."""

    def __init__(self, response=None, **_kwargs):
        self._response = response

    def with_structured_output(self, _schema):
        return FakeStructuredLLM(self._response)


class FakeToolCallingLLM:
    """Returns each response in order — one per .invoke() call — simulating the model first
    deciding to call a tool, then producing a final answer once given the tool's result."""

    def __init__(self, responses):
        self._responses = iter(responses)

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return next(self._responses)
