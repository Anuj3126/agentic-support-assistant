"""Resolves a logged-in name to a verified identity (customer or employee row), so an SQL-
enabled department agent scopes lookups to a real database record instead of trusting
whatever name happens to appear in the chat — which is exactly what previously let someone
say "I'm Priya" and then "I'm Jackson" in the same conversation and have both accepted. The
verified identity comes from the login (state["user_id"]), never from parsing the message.
"""

import sqlite3

from shopunow.config import BUSINESS_DB_PATH
from shopunow.db.schema import IDENTITY_KIND


def resolve_identity(department_id: str, name: str | None) -> dict | None:
    """Returns {"kind": "customer"|"employee", "id": int, "name": str}, or None if this
    department has no identity concept, or the logged-in name has no matching row."""
    kind = IDENTITY_KIND.get(department_id)
    if kind is None or not name:
        return None

    table, id_column = ("customers", "customer_id") if kind == "customer" else ("employees", "employee_id")
    uri = f"file:{BUSINESS_DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        row = conn.execute(f"SELECT {id_column} FROM {table} WHERE name = ?", (name,)).fetchone()
    finally:
        conn.close()

    return {"kind": kind, "id": row[0], "name": name} if row else None
