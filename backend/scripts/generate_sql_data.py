"""One-time data generator — NOT part of the runtime package, same as generate_qa.py.

Generates the synthetic dataset backing the SQL tool: customers/inventory/orders/
order_items/payments for Billing, Shipping, and Returns; employees/leave_balances/
it_tickets for HR and IT Support.

Five names are hand-seeded with a real, distinct scenario each — these are also the login
options in main.py / frontend/app.py, so logging in as any of them resolves to real data:
  - Priya Sharma (customer)   — a delivered jacket, ready to test the return-write action
  - Dev Gupta (customer)      — a genuine duplicate charge on one order
  - Neha Iyer (customer)      — an order still in transit (not yet delivered)
  - Asha Nair (employee)      — a low remaining PTO balance
  - Rohan Khan (employee)     — an open VPN ticket
Everyone else is randomly generated filler.

Run from backend/: `python scripts/generate_sql_data.py`
Writes shopunow/data/sql/*.json, which db/init_db.py then seeds into the SQLite db.
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

from shopunow.config import DATA_DIR
from shopunow.db.schema import LOGIN_NAMES

OUT_DIR = DATA_DIR / "sql"

random.seed(42)
START_DATE = date(2026, 1, 1)

FIRST_NAMES = ["Asha", "Ravi", "Maya", "Dev", "Arjun", "Neha", "Kabir", "Isha", "Rohan", "Tara"]
LAST_NAMES = ["Sharma", "Mehta", "Gupta", "Iyer", "Khan", "Nair", "Reddy", "Das", "Chopra", "Rao"]

INVENTORY = [
    # clothing
    {"name": "Classic Denim Jacket", "category": "clothing", "price": 2499.0},
    {"name": "Cotton Crew T-Shirt", "category": "clothing", "price": 499.0},
    {"name": "Slim Fit Chinos", "category": "clothing", "price": 1299.0},
    {"name": "Wool Blend Sweater", "category": "clothing", "price": 1899.0},
    {"name": "Rain Shell Jacket", "category": "clothing", "price": 2199.0},
    # DIY
    {"name": "Cordless Drill Kit", "category": "diy", "price": 3499.0},
    {"name": "Paint Roller Set", "category": "diy", "price": 599.0},
    {"name": "Garden Tool Set", "category": "diy", "price": 1199.0},
    {"name": "LED Work Light", "category": "diy", "price": 899.0},
    # books
    {"name": "The Midnight Library (Novel)", "category": "books", "price": 399.0},
    {"name": "Atomic Habits", "category": "books", "price": 449.0},
    {"name": "Kids' Picture Atlas", "category": "books", "price": 349.0},
    {"name": "DIY Home Repair Guide", "category": "books", "price": 499.0},
    # toys
    {"name": "Wooden Building Blocks", "category": "toys", "price": 799.0},
    {"name": "Remote Control Car", "category": "toys", "price": 1499.0},
    {"name": "Board Game: Strategy Quest", "category": "toys", "price": 999.0},
    {"name": "Plush Teddy Bear", "category": "toys", "price": 599.0},
    {"name": "1000-Piece Puzzle", "category": "toys", "price": 649.0},
]

ORDER_STATUSES = ["placed", "processing", "shipped", "delivered", "cancelled"]
PAYMENT_METHODS = ["credit_card", "debit_card", "upi", "net_banking"]
PAYMENT_STATUSES = ["success", "failed", "refunded", "pending"]
LEAVE_TYPES = ["PTO", "sick", "casual"]
IT_ISSUE_TYPES = ["vpn_access", "laptop_hardware", "software_install", "password_reset", "network"]
TICKET_STATUSES = ["open", "in_progress", "resolved", "closed"]
INTERNAL_DEPARTMENTS = ["hr", "it_support", "facilities"]


def random_date(days_span: int = 300) -> str:
    return (START_DATE + timedelta(days=random.randint(0, days_span))).isoformat()


def generate_inventory() -> list[dict]:
    return [
        {
            "item_id": i,
            "name": item["name"],
            "category": item["category"],
            "price": item["price"],
            "stock_qty": random.randint(5, 80),
        }
        for i, item in enumerate(INVENTORY, start=1)
    ]


def generate_customers(n: int = 24) -> list[dict]:
    seeded = [n["name"] for n in LOGIN_NAMES if n["kind"] == "customer"]
    customers = [
        {"customer_id": i + 1, "name": name, "email": f"{name.lower().replace(' ', '.')}@example.com"}
        for i, name in enumerate(seeded)
    ]
    for i in range(len(seeded) + 1, n + 1):
        first, last = random.choice(FIRST_NAMES), random.choice(LAST_NAMES)
        customers.append(
            {"customer_id": i, "name": f"{first} {last}", "email": f"{first.lower()}.{last.lower()}{i}@example.com"}
        )
    return customers


def _item_id(inventory: list[dict], name: str) -> int:
    return next(item["item_id"] for item in inventory if item["name"] == name)


def generate_orders_and_items(customers: list[dict], inventory: list[dict], n: int = 30):
    jacket = _item_id(inventory, "Classic Denim Jacket")
    drill = _item_id(inventory, "Cordless Drill Kit")
    rain_jacket = _item_id(inventory, "Rain Shell Jacket")

    orders, order_items = [], []
    order_item_id = 1

    def add_order(order_id, customer_id, status, days_ago, item_id, quantity=1):
        nonlocal order_item_id
        orders.append(
            {"order_id": order_id, "customer_id": customer_id, "order_date": random_date(days_span=days_ago), "status": status}
        )
        order_items.append(
            {
                "order_item_id": order_item_id,
                "order_id": order_id,
                "item_id": item_id,
                "quantity": quantity,
                "status": "active",
            }
        )
        order_item_id += 1

    # Priya Sharma (customer_id 1) — delivered jacket, ready for a return request
    add_order(1001, 1, "delivered", 5, jacket)
    # Dev Gupta (customer_id 2) — delivered drill, genuinely double-charged (see payments below)
    add_order(1002, 2, "delivered", 10, drill)
    # Neha Iyer (customer_id 3) — still in transit, not yet delivered
    add_order(1003, 3, "shipped", 2, rain_jacket)

    for order_id in range(1004, 1001 + n):
        customer = random.choice(customers)
        orders.append(
            {
                "order_id": order_id,
                "customer_id": customer["customer_id"],
                "order_date": random_date(),
                "status": random.choice(ORDER_STATUSES),
            }
        )
        for item in random.sample(inventory, k=random.randint(1, 3)):
            order_items.append(
                {
                    "order_item_id": order_item_id,
                    "order_id": order_id,
                    "item_id": item["item_id"],
                    "quantity": random.randint(1, 2),
                    "status": "active",
                }
            )
            order_item_id += 1

    return orders, order_items


def generate_payments(orders: list[dict], order_items: list[dict], inventory: list[dict]) -> list[dict]:
    price_by_item = {item["item_id"]: item["price"] for item in inventory}
    payments, payment_id = [], 5001

    def order_total(order_id: int) -> float:
        return round(
            sum(price_by_item[oi["item_id"]] * oi["quantity"] for oi in order_items if oi["order_id"] == order_id), 2
        )

    for order in orders:
        total = order_total(order["order_id"])
        payments.append(
            {
                "payment_id": payment_id,
                "order_id": order["order_id"],
                "amount": total,
                "method": random.choice(PAYMENT_METHODS),
                "payment_date": order["order_date"],
                "status": "success" if order["order_id"] in (1001, 1002, 1003) else random.choice(PAYMENT_STATUSES),
            }
        )
        payment_id += 1
        # Dev Gupta's order 1002 — a genuine accidental duplicate charge
        if order["order_id"] == 1002:
            payments.append(
                {
                    "payment_id": payment_id,
                    "order_id": 1002,
                    "amount": total,
                    "method": payments[-1]["method"],
                    "payment_date": order["order_date"],
                    "status": "success",
                }
            )
            payment_id += 1

    return payments


def generate_employees(n: int = 8) -> list[dict]:
    seeded = [n["name"] for n in LOGIN_NAMES if n["kind"] == "employee"]
    employees = [
        {"employee_id": i + 1, "name": name, "email": f"{name.lower().replace(' ', '.')}@shopunow.com", "department": "hr" if i == 0 else "it_support"}
        for i, name in enumerate(seeded)
    ]
    for i in range(len(seeded) + 1, n + 1):
        first, last = random.choice(FIRST_NAMES), random.choice(LAST_NAMES)
        employees.append(
            {
                "employee_id": i,
                "name": f"{first} {last}",
                "email": f"{first.lower()}.{last.lower()}{i}@shopunow.com",
                "department": random.choice(INTERNAL_DEPARTMENTS),
            }
        )
    return employees


def generate_leave_balances(employees: list[dict]) -> list[dict]:
    balances, balance_id = [], 3001
    for emp in employees:
        for leave_type in LEAVE_TYPES:
            # Asha Nair (employee_id 1) is deliberately low on PTO specifically
            if emp["employee_id"] == 1 and leave_type == "PTO":
                days = 1.5
            else:
                days = round(random.uniform(3, 18), 1)
            balances.append(
                {"leave_balance_id": balance_id, "employee_id": emp["employee_id"], "leave_type": leave_type, "balance_days": days}
            )
            balance_id += 1
    return balances


def generate_it_tickets(employees: list[dict], n: int = 20) -> list[dict]:
    tickets, ticket_id = [], 7001

    # Rohan Khan (employee_id 2) — an open VPN ticket to look up
    tickets.append(
        {
            "ticket_id": ticket_id,
            "employee_id": 2,
            "issue_type": "vpn_access",
            "status": "open",
            "created_date": random_date(days_span=3),
        }
    )
    ticket_id += 1

    for _ in range(n):
        emp = random.choice(employees)
        tickets.append(
            {
                "ticket_id": ticket_id,
                "employee_id": emp["employee_id"],
                "issue_type": random.choice(IT_ISSUE_TYPES),
                "status": random.choice(TICKET_STATUSES),
                "created_date": random_date(),
            }
        )
        ticket_id += 1
    return tickets


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = generate_inventory()
    customers = generate_customers()
    orders, order_items = generate_orders_and_items(customers, inventory)
    payments = generate_payments(orders, order_items, inventory)
    employees = generate_employees()
    leave_balances = generate_leave_balances(employees)
    it_tickets = generate_it_tickets(employees)

    for name, rows in [
        ("customers", customers),
        ("inventory", inventory),
        ("orders", orders),
        ("order_items", order_items),
        ("payments", payments),
        ("employees", employees),
        ("leave_balances", leave_balances),
        ("it_tickets", it_tickets),
    ]:
        out_path = OUT_DIR / f"{name}.json"
        out_path.write_text(json.dumps(rows, indent=2))
        print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
