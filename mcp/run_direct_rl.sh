#!/bin/bash
# Script to run the RL agent directly without going through the MCP server

# Exit on error
set -e

# Change to the script's directory
cd "$(dirname "$0")" || exit

echo "Setting up environment for direct RL agent..."

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

# Run the direct RL agent script
echo "Running direct RL agent..."
python direct_rl_agent.py "$@"

# Example usage:
# ./run_direct_rl.sh --environment simple_grid --episodes 10 --learning-rate 0.2