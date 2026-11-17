"""SQLite schema for ShopUNow's order/billing/shipping data (synthetic).

Scoped to the three external, order-related departments (Billing, Shipping, Returns) — HR/IT/
Facilities stay pure RAG, since there's no comparable transactional data behind them.
"""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inventory (
    item_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    stock_qty INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    item_id INTEGER NOT NULL REFERENCES inventory(item_id),
    quantity INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS payments (
    payment_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    amount REAL NOT NULL,
    method TEXT NOT NULL,
    payment_date TEXT NOT NULL,
    status TEXT NOT NULL
);
"""

# Every table/column the SQL guard knows about. Used to tell "unknown table" (typo, made up,
# or a system table like sqlite_master) apart from "a real table this department just isn't
# allowed to see."
ALLOWED_TABLES = {
    "customers": {"customer_id", "name", "email"},
    "inventory": {"item_id", "name", "category", "price", "stock_qty"},
    "orders": {"order_id", "customer_id", "order_date", "status"},
    "order_items": {"order_item_id", "order_id", "item_id", "quantity"},
    "payments": {"payment_id", "order_id", "amount", "method", "payment_date", "status"},
}

# Which of those tables each department's agent is allowed to query. A department not listed
# here gets no SQL tool at all.
DEPARTMENT_TABLES = {
    "billing": {"customers", "orders", "payments"},
    "shipping": {"customers", "orders", "order_items", "inventory"},
    "returns": {"customers", "orders", "order_items", "inventory"},
}
