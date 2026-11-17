"""Single source of truth for the departments ShopUNow's assistant supports.

Six departments total — the PDF asks for 4 minimum (2 internal, 2 external); the extra two
(Facilities, Returns) are Stretch Option 3 ("add more routes/departments").
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Department:
    id: str
    label: str
    audience: str  # "internal" or "external"
    description: str


DEPARTMENTS: dict[str, Department] = {
    d.id: d
    for d in [
        Department(
            id="hr",
            label="Human Resources",
            audience="internal",
            description=(
                "Handles employee lifecycle matters like leave/PTO, payroll, performance "
                "reviews, and company policy. Used by internal ShopUNow employees only."
            ),
        ),
        Department(
            id="it_support",
            label="IT Support",
            audience="internal",
            description=(
                "Manages technical issues related to company systems, hardware/software, "
                "VPN, and digital account access. Used by internal ShopUNow employees only."
            ),
        ),
        Department(
            id="facilities",
            label="Facilities & Admin",
            audience="internal",
            description=(
                "Supports employees with workspace logistics, building maintenance, access "
                "cards, and office supplies. Used by internal ShopUNow employees only."
            ),
        ),
        Department(
            id="billing",
            label="Billing & Payments",
            audience="external",
            description=(
                "Assists customers with invoice issues, refunds, payment methods, and "
                "overcharges. Used by external ShopUNow customers only."
            ),
        ),
        Department(
            id="shipping",
            label="Shipping & Delivery",
            audience="external",
            description=(
                "Handles customer queries about order tracking, delivery delays, damaged "
                "goods, and return pickups. Used by external ShopUNow customers only."
            ),
        ),
        Department(
            id="returns",
            label="Returns & Exchanges",
            audience="external",
            description=(
                "Helps customers initiate returns or exchanges, checks eligibility windows, "
                "and explains the refund timeline. Used by external ShopUNow customers only."
            ),
        ),
    ]
}

DEPARTMENT_IDS: list[str] = list(DEPARTMENTS.keys())
