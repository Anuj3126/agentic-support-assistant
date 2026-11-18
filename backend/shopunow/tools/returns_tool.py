"""A narrow, dedicated action for Returns — deliberately NOT free-form SQL. The model only
ever supplies which order/item and why; the customer identity is closed over from the
verified login (never a model-supplied argument, so it can't be spoofed by anything said in
the chat). "Genuine" is a check against the actual database state, not something the model
decides on its own — a request is rejected unless the order actually belongs to this
customer, actually contains this item, and is actually delivered.
"""

import sqlite3

from langchain_core.tools import tool

from shopunow.config import BUSINESS_DB_PATH


def request_return(customer_id: int, order_id: int, item_id: int, reason: str) -> dict:
    conn = sqlite3.connect(BUSINESS_DB_PATH)
    try:
        order = conn.execute(
            "SELECT status FROM orders WHERE order_id = ? AND customer_id = ?", (order_id, customer_id)
        ).fetchone()
        if order is None:
            return {"success": False, "error": "no such order for this account"}
        if order[0] != "delivered":
            return {"success": False, "error": f"order status is '{order[0]}' — only delivered orders can be returned"}

        item = conn.execute(
            "SELECT order_item_id, status FROM order_items WHERE order_id = ? AND item_id = ?",
            (order_id, item_id),
        ).fetchone()
        if item is None:
            return {"success": False, "error": "that item is not part of this order"}
        order_item_id, item_status = item
        if item_status != "active":
            return {"success": False, "error": f"this item already has return status '{item_status}'"}

        conn.execute("UPDATE order_items SET status = 'return_requested' WHERE order_item_id = ?", (order_item_id,))
        conn.commit()
        return {
            "success": True,
            "order_id": order_id,
            "item_id": item_id,
            "new_status": "return_requested",
            "reason": reason,
        }
    finally:
        conn.close()


def make_return_tool(customer_id: int):
    """Builds a @tool closed over one verified customer's id — the model can never supply
    or override whose account this acts on."""

    def _request_return(order_id: int, item_id: int, reason: str) -> dict:
        return request_return(customer_id, order_id, item_id, reason)

    _request_return.__name__ = "request_return"
    _request_return.__doc__ = (
        "Submit a return request for one item on one of this customer's own orders. Only "
        "succeeds for orders that are actually delivered and actually contain that item. "
        "Use the read-only lookup tool first if you need to confirm which order/item number "
        "the customer means before calling this."
    )
    return tool(_request_return)
