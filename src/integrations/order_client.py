"""
src/integrations/order_client.py
==================================
Thin MCP client wrapper — calls the order-management-app's MCP server
(mcp_server/server.py, expected running at http://127.0.0.1:9100/mcp).

Bridges FastMCP's async Client into a plain sync function, the same
asyncio.run() pattern your existing tool files (doctor_tool.py,
medicine_tool.py) already use to call their own async retrievers.
"""

import os

from fastmcp import Client

from src.logger import get_logger

logger = get_logger(__name__)

ORDER_MCP_URL = os.getenv("ORDER_MCP_URL", "http://127.0.0.1:9100/mcp")


async def _call_tool_async(tool_name: str, **kwargs) -> dict:
    async with Client(ORDER_MCP_URL) as client:
        result = await client.call_tool(tool_name, kwargs)
        if result.data is not None:
            return result.data
        # Fallback if the server didn't return structured data for some reason
        return {"message": str(result.content)}


def call_order_tool(tool_name: str, **kwargs) -> dict:
    """
    Sync wrapper. Returns the tool's dict result, e.g.
    {"status": "submitted", "order_id": 4, "message": "..."}

    If the order-management-app isn't running, this raises — callers should
    catch and turn it into a user-facing "ordering is unavailable right now"
    message rather than letting the whole request 500.
    """
    import asyncio
    try:
        return asyncio.run(_call_tool_async(tool_name, **kwargs))
    except Exception as e:
        logger.warning("MCP call to '%s' failed: %s", tool_name, e)
        return {
            "status": "error",
            "message": (
                "Sorry, I couldn't reach the ordering system right now. "
                "Please make sure it's running, or try again shortly."
            ),
        }