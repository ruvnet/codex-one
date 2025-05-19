from types import SimpleNamespace
from .mcp_types import TextContent
from .server.fastmcp import FastMCP

class types:
    TextContent = TextContent

class server:
    fastmcp = SimpleNamespace(FastMCP=FastMCP)
