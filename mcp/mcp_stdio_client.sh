#!/bin/bash
# Simple client script to interact with MCP server in stdio mode

# Usage: echo '{"tool":"echo","params":{"text":"hello","transform":"upper"}}' | ./mcp_stdio_client.sh

# Change to the script's directory
cd "$(dirname "$0")" || exit

# Check if the venv exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run ./run_mcp_stdio.sh first to set up the environment." >&2
    exit 1
fi

# Activate the virtual environment
source venv/bin/activate

# Read JSON from stdin
echo "Sending request to MCP server..."
REQUEST=$(cat)

# Pass the JSON as a command-line argument to dspy_mcp-client
dspy_mcp-client "$REQUEST"

# Example usage:
# echo '{"tool":"echo","params":{"text":"hello","transform":"upper"}}' | ./mcp_stdio_client.sh
# echo '{"tool":"dspy_echo","params":{"text":"testing dspy agent","transform":"lower"}}' | ./mcp_stdio_client.sh