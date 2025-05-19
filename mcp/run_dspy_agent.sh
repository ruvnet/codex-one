#!/bin/bash
# Script to set up and run the DSPy MCP server and client

# Exit on error
set -e

echo "Setting up DSPy MCP environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

# Run the server in background using SSE transport
echo "Starting MCP server in SSE mode on port 3001..."
dspy_mcp-server --transport sse --port 3001 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 2

# Show available tools using curl
echo "Showing available tools:"
curl -s http://localhost:3001/tools | python -m json.tool

echo ""
echo "Testing the 'echo' tool using curl:"
curl -s -X POST http://localhost:3001/tools/echo \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello from DSPy MCP!", "transform": "upper"}' | python -m json.tool

echo ""
echo "Testing the 'dspy_echo' tool (using DSPy agent pipeline):"
curl -s -X POST http://localhost:3001/tools/dspy_echo \
    -H "Content-Type: application/json" \
    -d '{"text": "Testing the DSPy agent pipeline!", "transform": "upper"}' | python -m json.tool

# Clean up - kill the server process
echo ""
echo "Tests complete. Shutting down server..."
kill $SERVER_PID

echo ""
echo "You can run the server manually with: dspy_mcp-server --transport sse"
echo "You can test the tools with: curl -X POST http://localhost:3001/tools/dspy_echo -H 'Content-Type: application/json' -d '{\"text\": \"Your message\"}'"
echo "You can also use the client with: dspy_mcp-client \"Your message\" --transform upper"

echo ""
echo "To use the DSPy agent in the Python code:"
echo "from dspy_mcp.pipeline.agent_pipeline import run_agent"
echo "result = run_agent(text=\"Your message\", transform=\"upper\")"
echo "print(result)"