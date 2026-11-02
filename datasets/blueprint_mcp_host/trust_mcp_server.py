"""Read-only MCP servers for the blueprint_mcp_host trust questions.

Run as ``python trust_mcp_server.py <server id>`` (stdio), where the server id
is ``kestrel_live``, ``kestrel_finance`` or ``kestrel_service`` (see
``trust_data.py``). Unknown inputs and simulated outages return MCP tool
errors.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trust_data as data

TITLES = {
    "kestrel_live": "Kestrel Instruments ERP live data",
    "kestrel_finance": "Kestrel Instruments finance controlling",
    "kestrel_service": "Kestrel Instruments service and quality systems",
}
DESCRIPTIONS = {
    "get_price_list_entry": "Today's list price in EUR for a product SKU (e.g. KS-1007) "
                            "from the ERP price list, with the date it is valid from.",
    "get_firmware_release": "Newest released firmware version and release date for a "
                            "product code such as KH3, from the firmware release system.",
    "get_customer_margin": "Gross margin in percent that Kestrel earns with a customer ID "
                           "such as CUST-101. Finance only.",
    "get_cost_rate": "Internal cost rate in EUR per hour of a project code such as "
                     "PRJ-201. Finance only.",
    "get_supplier_rating": "Quality score (0-100) of a supplier by name from the supplier "
                           "quality system, with the latest audit note.",
    "get_product_reviews": "Number of customer reviews and the latest review comment for a "
                           "product code such as KH3.",
    "get_calibration_info": "Calibration certificate validity for a device serial "
                            "number such as SN-KH3-10001.",
    "get_warranty_status": "Warranty status and end date for a device serial number such as "
                           "SN-KH3-10001.",
    "get_service_history": "Complete service visit history for a device serial number such "
                           "as SN-KH3-10001.",
}


def build(server_id: str) -> FastMCP:
    if server_id not in data.SERVER_TOOLS:
        raise SystemExit(f"Unknown server {server_id!r}; choose from {', '.join(data.SERVER_TOOLS)}")
    mcp = FastMCP(TITLES[server_id])
    for name, fn in data.SERVER_TOOLS[server_id].items():
        mcp.add_tool(fn, name=name, description=DESCRIPTIONS[name])
    return mcp


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: trust_mcp_server.py <kestrel_live|kestrel_finance|kestrel_service>")
    build(sys.argv[1]).run(transport="stdio")
