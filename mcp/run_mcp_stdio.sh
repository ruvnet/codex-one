#!/bin/bash
# Script to run the MCP server in stdio mode with DSPy Agent

# Exit on error
set -e

# Change to the script's directory
cd "$(dirname "$0")" || exit

echo "Setting up DSPy MCP environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if ! pip show dspy &> /dev/null; then
    echo "Installing dependencies..."
    pip install dspy dspy-ai
fi

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

# Run the server with stdio transport
echo "Starting MCP server in stdio mode..."
PYTHONPATH="$(pwd)" dspy_mcp-server --transport stdio

# The script will pass through stdin/stdout directly to the MCP server