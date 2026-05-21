"""Demo tools for the Langfuse tracing showcase.

These are intentionally simple, business-recognisable functions
(order lookup, product catalog, discount math) so non-technical
viewers can follow what the agent decides to call and why.
"""
from __future__ import annotations

from datetime import date


# --- Mock data ---------------------------------------------------------------

_ORDERS = {
    "ORD-1001": {"status": "Shipped",   "carrier": "FedEx",  "eta": "2026-05-23"},
    "ORD-1002": {"status": "Processing","carrier": None,     "eta": "2026-05-27"},
    "ORD-1003": {"status": "Delivered", "carrier": "UPS",    "eta": "2026-05-19"},
    "ORD-1004": {"status": "Cancelled", "carrier": None,     "eta": None},
}

_PRODUCTS = {
    "laptop":   {"sku": "LP-220", "price_usd": 1299.00, "in_stock": 18},
    "monitor":  {"sku": "MN-027", "price_usd": 349.50,  "in_stock": 42},
    "keyboard": {"sku": "KB-014", "price_usd": 89.99,   "in_stock": 0},
    "mouse":    {"sku": "MS-009", "price_usd": 29.99,   "in_stock": 134},
}


# --- Tool implementations ----------------------------------------------------

def get_order_status(order_id: str) -> dict:
    order = _ORDERS.get(order_id.upper().strip())
    if not order:
        return {"error": f"No order found with id '{order_id}'."}
    return {"order_id": order_id.upper().strip(), **order}


def lookup_product(product_name: str) -> dict:
    item = _PRODUCTS.get(product_name.lower().strip())
    if not item:
        return {
            "error": f"No product matching '{product_name}'.",
            "available": sorted(_PRODUCTS.keys()),
        }
    return {"product": product_name.lower().strip(), **item}


def calculate_discount(price_usd: float, discount_percent: float) -> dict:
    if price_usd < 0 or discount_percent < 0 or discount_percent > 100:
        return {"error": "Price must be >= 0 and discount between 0 and 100."}
    saved = round(price_usd * discount_percent / 100, 2)
    final = round(price_usd - saved, 2)
    return {
        "original_price_usd": price_usd,
        "discount_percent": discount_percent,
        "amount_saved_usd": saved,
        "final_price_usd": final,
    }


def get_today() -> dict:
    return {"today": date.today().isoformat()}


# --- Tool schema exposed to Claude ------------------------------------------

TOOL_SCHEMA = [
    {
        "name": "get_order_status",
        "description": "Look up the shipping status of a customer order by its order id (e.g. ORD-1001).",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order identifier, like ORD-1001."},
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "lookup_product",
        "description": "Look up product price, SKU and stock level by product name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "Product name, e.g. 'laptop'."},
            },
            "required": ["product_name"],
        },
    },
    {
        "name": "calculate_discount",
        "description": "Compute final price after applying a percentage discount.",
        "input_schema": {
            "type": "object",
            "properties": {
                "price_usd": {"type": "number"},
                "discount_percent": {"type": "number", "description": "0 to 100."},
            },
            "required": ["price_usd", "discount_percent"],
        },
    },
    {
        "name": "get_today",
        "description": "Return today's date in ISO format. Use when the user asks about 'today' or relative dates.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


TOOL_REGISTRY = {
    "get_order_status": get_order_status,
    "lookup_product": lookup_product,
    "calculate_discount": calculate_discount,
    "get_today": get_today,
}
