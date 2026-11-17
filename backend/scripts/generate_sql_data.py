"""One-time data generator — NOT part of the runtime package, same as generate_qa.py.

Generates a small, referentially-consistent synthetic dataset backing the Billing, Shipping,
and Returns departments' SQL tool: customers, an inventory catalog, orders, order line items,
and payments. One customer ("Priya Sharma") is hand-seeded with a real order containing a
jacket, since she's been the running example throughout this project's testing — everyone
else is randomly generated filler.

Run from backend/: `python scripts/generate_sql_data.py`
Writes shopunow/data/sql/*.json, which db/init_db.py then seeds into the SQLite db.
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

from shopunow.config import DATA_DIR

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
    customers = [{"customer_id": 1, "name": "Priya Sharma", "email": "priya.sharma1@example.com"}]
    for i in range(2, n + 1):
        first, last = random.choice(FIRST_NAMES), random.choice(LAST_NAMES)
        customers.append(
            {"customer_id": i, "name": f"{first} {last}", "email": f"{first.lower()}.{last.lower()}{i}@example.com"}
        )
    return customers


def generate_orders_and_items(customers: list[dict], inventory: list[dict], n: int = 30):
    jacket = next(item for item in inventory if item["name"] == "Classic Denim Jacket")
    orders, order_items = [], []
    order_item_id = 1

    # Priya's seeded order: a delivered jacket, returnable within the standard window
    orders.append(
        {"order_id": 1001, "customer_id": 1, "order_date": random_date(days_span=5), "status": "delivered"}
    )
    order_items.append({"order_item_id": order_item_id, "order_id": 1001, "item_id": jacket["item_id"], "quantity": 1})
    order_item_id += 1

    for order_id in range(1002, 1001 + n):
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
                {"order_item_id": order_item_id, "order_id": order_id, "item_id": item["item_id"], "quantity": random.randint(1, 2)}
            )
            order_item_id += 1

    return orders, order_items


def generate_payments(orders: list[dict], order_items: list[dict], inventory: list[dict]) -> list[dict]:
    price_by_item = {item["item_id"]: item["price"] for item in inventory}
    payments, payment_id = [], 5001
    for order in orders:
        total = sum(
            price_by_item[oi["item_id"]] * oi["quantity"]
            for oi in order_items
            if oi["order_id"] == order["order_id"]
        )
        payments.append(
            {
                "payment_id": payment_id,
                "order_id": order["order_id"],
                "amount": round(total, 2),
                "method": random.choice(PAYMENT_METHODS),
                "payment_date": order["order_date"],
                "status": "success" if order["order_id"] == 1001 else random.choice(PAYMENT_STATUSES),
            }
        )
        payment_id += 1
    return payments


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = generate_inventory()
    customers = generate_customers()
    orders, order_items = generate_orders_and_items(customers, inventory)
    payments = generate_payments(orders, order_items, inventory)

    for name, rows in [
        ("customers", customers),
        ("inventory", inventory),
        ("orders", orders),
        ("order_items", order_items),
        ("payments", payments),
    ]:
        out_path = OUT_DIR / f"{name}.json"
        out_path.write_text(json.dumps(rows, indent=2))
        print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
