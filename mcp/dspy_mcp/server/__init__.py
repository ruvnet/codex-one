"""MCP server package initialization"""

from dspy_mcp.config import load_config
from dspy_mcp.server.app import create_mcp_server

# Create server instance with default configuration
server = create_mcp_server(load_config())

__all__ = ["server", "create_mcp_server"]
