"""MCP server implementation with Echo tool and RL agent"""

import asyncio
import click
import sys
from typing import Optional

from mcp import types
from mcp.server.fastmcp import FastMCP

from dspy_mcp.config import ServerConfig, load_config
from dspy_mcp.logging_config import setup_logging, logger
from dspy_mcp.tools.echo import echo
from dspy_mcp.pipeline.agent_pipeline import run_agent
from dspy_mcp.pipeline.rl_agent import RLAgent


def create_mcp_server(config: Optional[ServerConfig] = None) -> FastMCP:
    """Create and configure the MCP server instance"""
    if config is None:
        config = load_config()
    
    # Set up logging first
    setup_logging(config)
    
    server = FastMCP(config.name)

    # Register all tools with the server
    register_tools(server)

    return server


def register_tools(mcp_server: FastMCP) -> None:
    """Register all MCP tools with the server"""

    @mcp_server.tool(
        name="echo",
        description="Echo back the input text with optional case transformation",
    )
    def echo_tool(text: str, transform: Optional[str] = None) -> types.TextContent:
        """Wrapper around the echo tool implementation"""
        return echo(text, transform)

    @mcp_server.tool(
        name="dspy_echo",
        description="Echo text using the DSPy pipeline",
    )
    def echo_agent_tool(text: str, transform: Optional[str] = None) -> types.TextContent:
        """Echo using the DSPy agent pipeline."""
        result = run_agent(text=text, transform=transform)
        return types.TextContent(type="text", text=result, format="text/plain")
    
    @mcp_server.tool(
        name="dspy_rl_agent",
        description="Run reinforcement learning agent on specified environment",
    )
    def rl_agent_tool(
        environment: str,
        episodes: int = 10,
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1
    ) -> types.TextContent:
        """Run the reinforcement learning agent."""
        # Initialize the RL agent
        rl_agent = RLAgent()
        
        # Run the agent
        logger.info(f"Running RL agent with environment={environment}, episodes={episodes}")
        prediction = rl_agent.forward(
            environment=environment,
            episodes=episodes,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon
        )
        
        return types.TextContent(type="text", text=prediction.result, format="text/plain")


# Create a server instance that can be imported by the MCP CLI
server = create_mcp_server()


@click.command()
@click.option("--port", default=3001, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type (stdio or sse)",
)
def main(port: int, transport: str) -> int:
    """Run the server with specified transport."""
    try:
        if transport == "stdio":
            asyncio.run(server.run_stdio_async())
        else:
            server.settings.port = port
            asyncio.run(server.run_sse_async())
        return 0
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())