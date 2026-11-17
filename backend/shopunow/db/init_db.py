"""Creates and seeds the business-data SQLite db from the synthetic JSON fixtures.

Run from backend/: `python -m shopunow.db.init_db`
Idempotent — skips seeding if the customers table already has rows.
"""

import json
import sqlite3
from pathlib import Path

from shopunow.config import BUSINESS_DB_PATH, DATA_DIR
from shopunow.db.schema import SCHEMA_SQL

SQL_DATA_DIR = DATA_DIR / "sql"

TABLE_COLUMNS = {
    "customers": ["customer_id", "name", "email"],
    "inventory": ["item_id", "name", "category", "price", "stock_qty"],
    "orders": ["order_id", "customer_id", "order_date", "status"],
    "order_items": ["order_item_id", "order_id", "item_id", "quantity"],
    "payments": ["payment_id", "order_id", "amount", "method", "payment_date", "status"],
}

# Parent tables must be seeded before the tables that reference them.
SEED_ORDER = ["customers", "inventory", "orders", "order_items", "payments"]


def _load_rows(table: str) -> list[dict]:
    path = SQL_DATA_DIR / f"{table}.json"
    if not path.exists():
        raise RuntimeError(f"missing {path} — run `python scripts/generate_sql_data.py` first")
    return json.loads(path.read_text())


def seed(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for table in SEED_ORDER:
        cols = TABLE_COLUMNS[table]
        placeholders = ", ".join("?" * len(cols))
        for row in _load_rows(table):
            cur.execute(
                f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
                [row[c] for c in cols],
            )
    conn.commit()


def init_db(db_path: Path = BUSINESS_DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        already_seeded = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] > 0
        if already_seeded:
            print(f"{db_path} already seeded, skipping")
            return
        seed(conn)
        print(f"seeded {db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
