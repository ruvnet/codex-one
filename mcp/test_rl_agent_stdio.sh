#!/bin/bash
# Script to test running the RL agent in stdio mode

# Exit on error
set -e

# Change to the script's directory
cd "$(dirname "$0")" || exit

# Test input for the RL agent
TEST_INPUT='{"text": "Testing simple echo", "transform": "upper"}'

# Test input for the RL agent - directly to dspy_mcp-client
TEST_RL_INPUT='{
  "tool": "dspy_rl_agent",
  "params": {
    "environment": "simple_grid",
    "episodes": 5,
    "learning_rate": 0.1,
    "gamma": 0.9,
    "epsilon": 0.1
  }
}'

echo "Testing echo in stdio mode..."
echo "Sending input: $TEST_INPUT"

# Send the echo test input to the stdio client
echo "$TEST_INPUT" | ./mcp_stdio_client.sh

echo
echo "Testing RL agent in stdio mode..."
echo "Sending input: $TEST_RL_INPUT"

# Send the RL agent test input to the stdio client
dspy_mcp-client "$TEST_RL_INPUT"

echo
echo "Note: If the test fails, make sure the RL agent is properly implemented"
echo "and registered in the MCP server. Check dspy_mcp/pipeline/agent_pipeline.py"
echo "and dspy_mcp/server/app.py for details."