"""Logging configuration for dspy-mcp MCP server"""

import logging
import logging.handlers
import os
import platform
import sys
from pathlib import Path
from typing import Optional

from .config import ServerConfig


def get_default_log_dir() -> Path:
    """Get the default log directory based on OS standards"""
    system = platform.system().lower()

    if system == "darwin":  # macOS
        return Path.home() / "Library" / "Logs" / "mcp-servers"
    elif system == "linux":
        if os.geteuid() == 0:
            return Path("/var/log/mcp-servers")
        else:
            return Path.home() / ".local" / "state" / "mcp-servers" / "logs"
    elif system == "windows":
        return Path.home() / "AppData" / "Local" / "mcp-servers" / "logs"
    else:
        return Path.home() / ".mcp-servers" / "logs"


def setup_logging(server_config: Optional[ServerConfig] = None) -> None:
    """Configure logging for the dspy-mcp MCP server"""
    # Get log level from server config, environment, or default to INFO
    log_level = "INFO"
    if server_config and hasattr(server_config, "log_level"):
        log_level = server_config.log_level.upper()
    elif os.getenv("LOG_LEVEL"):
        log_level = os.getenv("LOG_LEVEL").upper()

    valid_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    effective_level = valid_levels.get(log_level, logging.INFO)

    # Configure basic logging first
    logging.basicConfig(
        level=effective_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[],  # We'll add handlers explicitly
    )

    # Get all relevant loggers
    project_logger = logging.getLogger("dspy_mcp")
    mcp_logger = logging.getLogger("mcp")  # Capture all MCP logs
    mcp_server_logger = logging.getLogger("mcp.server")
    uvicorn_logger = logging.getLogger("uvicorn")
    root_logger = logging.getLogger()

    # Remove any existing handlers to avoid duplicates
    for logger in [
        project_logger,
        mcp_logger,
        mcp_server_logger,
        uvicorn_logger,
        root_logger,
    ]:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(effective_level)
    stderr_handler.setFormatter(formatter)
    root_logger.addHandler(stderr_handler)

    # Set up file logging
    log_path = get_default_log_dir()
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "dspy_mcp.log"

    # Add rotating file handler (10MB files, keep 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(effective_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Set levels for all loggers
    for logger in [
        project_logger,
        mcp_logger,
        mcp_server_logger,
        uvicorn_logger,
    ]:
        logger.setLevel(effective_level)
        # Ensure propagation is enabled
        logger.propagate = True


# Export the logger for use in other modules
logger = logging.getLogger("dspy_mcp")
