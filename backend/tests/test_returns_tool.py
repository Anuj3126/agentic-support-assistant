from shopunow.tools import returns_tool


def test_genuine_return_succeeds_and_updates_the_database(seeded_sqlite_db, monkeypatch):
    monkeypatch.setattr(returns_tool, "BUSINESS_DB_PATH", seeded_sqlite_db)

    result = returns_tool.request_return(customer_id=1, order_id=1001, item_id=1, reason="doesn't fit")

    assert result == {
        "success": True,
        "order_id": 1001,
        "item_id": 1,
        "new_status": "return_requested",
        "reason": "doesn't fit",
    }

    import sqlite3

    conn = sqlite3.connect(seeded_sqlite_db)
    status = conn.execute("SELECT status FROM order_items WHERE order_item_id = 1").fetchone()[0]
    conn.close()
    assert status == "return_requested"


def test_a_customer_cannot_return_someone_elses_order(seeded_sqlite_db, monkeypatch):
    monkeypatch.setattr(returns_tool, "BUSINESS_DB_PATH", seeded_sqlite_db)

    # order 1001 belongs to customer_id=1 (Priya), not customer_id=2 (Dev)
    result = returns_tool.request_return(customer_id=2, order_id=1001, item_id=1, reason="not mine")

    assert result == {"success": False, "error": "no such order for this account"}


def test_an_order_not_yet_delivered_cannot_be_returned(seeded_sqlite_db, monkeypatch):
    monkeypatch.setattr(returns_tool, "BUSINESS_DB_PATH", seeded_sqlite_db)

    # order 1003 belongs to customer_id=1 but is still "shipped", not "delivered"
    result = returns_tool.request_return(customer_id=1, order_id=1003, item_id=1, reason="too soon")

    assert result["success"] is False
    assert "only delivered orders" in result["error"]


def test_requesting_a_return_twice_is_rejected_the_second_time(seeded_sqlite_db, monkeypatch):
    monkeypatch.setattr(returns_tool, "BUSINESS_DB_PATH", seeded_sqlite_db)

    first = returns_tool.request_return(customer_id=1, order_id=1001, item_id=1, reason="first attempt")
    second = returns_tool.request_return(customer_id=1, order_id=1001, item_id=1, reason="second attempt")

    assert first["success"] is True
    assert second["success"] is False
    assert "already has return status" in second["error"]


def test_make_return_tool_closes_over_the_customer_id_not_a_model_supplied_argument(seeded_sqlite_db, monkeypatch):
    """The model only ever supplies order_id/item_id/reason — never whose account it acts
    on. This confirms the built tool's signature genuinely has no customer_id parameter."""
    monkeypatch.setattr(returns_tool, "BUSINESS_DB_PATH", seeded_sqlite_db)

    tool = returns_tool.make_return_tool(customer_id=1)
    assert "customer_id" not in tool.args

    result = tool.invoke({"order_id": 1001, "item_id": 1, "reason": "testing the bound tool"})
    assert result["success"] is True
