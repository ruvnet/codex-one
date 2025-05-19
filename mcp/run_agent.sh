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
. venv/bin/activate

# Install dependencies and the package in development mode
echo "Installing dependencies..."
pip install dspy dspy-ai
echo "Installing package in development mode..."
pip install -e .

# Run the server in background using SSE transport (with verbose output)
echo "Starting MCP server in SSE mode on port 3001..."
PYTHONPATH=/workspaces/codex-one/mcp dspy_mcp-server --transport sse --port 3001 &
SERVER_PID=$!

# Register cleanup function to kill server on exit
cleanup() {
    echo "Shutting down server..."
    kill $SERVER_PID 2>/dev/null || true
    exit 0
}

# Set up trap to catch Ctrl+C and other termination signals
trap cleanup SIGINT SIGTERM EXIT

# Wait for server to start
echo "Waiting for server to start..."
sleep 5

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

# Loop for interactive testing
echo ""
echo "Interactive agent testing mode. Enter messages to send to the agent (Ctrl+C to exit)"
echo "Format: [text] or [text]:[transform]"
echo "Example: Hello, world! or Hello, world!:upper"

while true; do
    echo -n "> "
    read input
    
    # Check if input contains a transform
    if [[ "$input" == *":"* ]]; then
        text="${input%:*}"
        transform="${input##*:}"
    else
        text="$input"
        transform="null"
    fi
    
    # Exit if input is 'exit' or 'quit'
    if [[ "$text" == "exit" || "$text" == "quit" ]]; then
        break
    fi
    
    # Format JSON payload based on transform value
    if [[ "$transform" == "null" ]]; then
        payload="{\"text\": \"$text\"}"
    else
        payload="{\"text\": \"$text\", \"transform\": \"$transform\"}"
    fi
    
    # Send request to dspy_echo tool
    echo "Calling DSPy agent with: $text (transform: ${transform})"
    curl -s -X POST http://localhost:3001/tools/dspy_echo \
        -H "Content-Type: application/json" \
        -d "$payload" | python -m json.tool
done

# Cleanup is handled by the trap