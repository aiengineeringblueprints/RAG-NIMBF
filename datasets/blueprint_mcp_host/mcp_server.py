"""Read-only MCP fixture server for the blueprint_mcp_host dataset.

Run as ``python mcp_server.py`` (stdio). Every tool serves synthetic data the
answering model cannot know, except ``get_weather``, a decoy no question
needs. Unknown inputs return MCP tool errors.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixture_data as data

mcp = FastMCP("Kestrel Instruments business systems")


@mcp.tool()
def search_directory(query: str) -> list[dict]:
    """Search the employee directory by name, job title, department or office.
    Returns title, department, office and phone extension."""
    return data.search_directory(query)


@mcp.tool()
def get_order_status(order_id: str) -> dict:
    """Look up a customer order (e.g. ORD-1001): product, SKU, quantity, status,
    carrier, date, shipping weight in kg and order value in EUR."""
    return data.get_order_status(order_id)


@mcp.tool()
def get_stock_level(sku: str) -> dict:
    """Current warehouse stock for a product SKU (e.g. KS-1007), with the
    expected restock date when out of stock."""
    return data.get_stock_level(sku)


@mcp.tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between EUR, USD, NOK, SEK, DKK, GBP, CHF, PLN and JPY using today's
    company exchange rates."""
    return data.convert_currency(amount, from_currency, to_currency)


@mcp.tool()
def count_open_tickets(product_code: str) -> dict:
    """Number of open support tickets and the most common issue for a product
    code such as KH3 or KF5."""
    return data.count_open_tickets(product_code)


@mcp.tool()
def get_shipping_quote(weight_kg: float, destination_country: str) -> dict:
    """Shipping price in EUR from the Stavanger warehouse for a parcel weight in
    kg to a destination country code (NO, SE, DK, FI, DE, NL, FR, PL, UK, US, CA, JP)."""
    return data.get_shipping_quote(weight_kg, destination_country)


@mcp.tool()
def get_project_status(project_code: str) -> dict:
    """Live status of a customer project (e.g. PRJ-201): budget spent in EUR,
    percent complete, hours booked and risk level."""
    return data.get_project_status(project_code)


@mcp.tool()
def get_invoice_balance(customer_id: str) -> dict:
    """Outstanding invoice balance in EUR and number of overdue invoices for a
    customer ID such as CUST-101."""
    return data.get_invoice_balance(customer_id)


@mcp.tool()
def get_weather(city: str) -> dict:
    """Current weather forecast for a city."""
    return data.get_weather(city)


@mcp.tool()
def get_stock_price(ticker: str) -> dict:
    """Latest share price for a stock ticker symbol."""
    return data.get_stock_price(ticker)


if __name__ == "__main__":
    mcp.run(transport="stdio")
