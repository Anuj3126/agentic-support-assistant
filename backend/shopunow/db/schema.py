"""SQLite schema for ShopUNow's business data (synthetic): orders/billing/shipping for
Billing/Shipping/Returns, and employee records for HR/IT Support.
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
    quantity INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS payments (
    payment_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    amount REAL NOT NULL,
    method TEXT NOT NULL,
    payment_date TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS employees (
    employee_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    department TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS leave_balances (
    leave_balance_id INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL REFERENCES employees(employee_id),
    leave_type TEXT NOT NULL,
    balance_days REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS it_tickets (
    ticket_id INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL REFERENCES employees(employee_id),
    issue_type TEXT NOT NULL,
    status TEXT NOT NULL,
    created_date TEXT NOT NULL
);
"""

# Every table/column the SQL guard knows about. Used to tell "unknown table" (typo, made up,
# or a system table like sqlite_master) apart from "a real table this department just isn't
# allowed to see."
ALLOWED_TABLES = {
    "customers": {"customer_id", "name", "email"},
    "inventory": {"item_id", "name", "category", "price", "stock_qty"},
    "orders": {"order_id", "customer_id", "order_date", "status"},
    "order_items": {"order_item_id", "order_id", "item_id", "quantity", "status"},
    "payments": {"payment_id", "order_id", "amount", "method", "payment_date", "status"},
    "employees": {"employee_id", "name", "email", "department"},
    "leave_balances": {"leave_balance_id", "employee_id", "leave_type", "balance_days"},
    "it_tickets": {"ticket_id", "employee_id", "issue_type", "status", "created_date"},
}

# Which of those tables each department's agent is allowed to read. A department not listed
# here gets no SQL tool at all. (Returns also gets a separate, narrow write action — see
# tools/returns_tool.py — which isn't part of this read-only whitelist.)
DEPARTMENT_TABLES = {
    "billing": {"customers", "orders", "payments"},
    "shipping": {"customers", "orders", "order_items", "inventory"},
    "returns": {"customers", "orders", "order_items", "inventory"},
    "hr": {"employees", "leave_balances"},
    "it_support": {"employees", "it_tickets"},
}

# Which table a department's identity check resolves against — "customer" departments look
# the logged-in name up in `customers`, "employee" departments look it up in `employees`.
IDENTITY_KIND = {
    "billing": "customer",
    "shipping": "customer",
    "returns": "customer",
    "hr": "employee",
    "it_support": "employee",
}

# The 5 login options (see scripts/generate_sql_data.py for the scenario each one backs).
LOGIN_NAMES = [
    {"name": "Priya Sharma", "kind": "customer"},
    {"name": "Dev Gupta", "kind": "customer"},
    {"name": "Neha Iyer", "kind": "customer"},
    {"name": "Asha Nair", "kind": "employee"},
    {"name": "Rohan Khan", "kind": "employee"},
]
