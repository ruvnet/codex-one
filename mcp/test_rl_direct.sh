#!/bin/bash
# Direct test script for RL agent using stdio transport

# Change to the script's directory
cd "$(dirname "$0")" || exit

# Make sure virtual environment is activated
if [ -d "venv" ]; then
    . venv/bin/activate
fi

echo "Testing RL agent directly..."

# Run the direct RL agent with fewer episodes for quick testing
./run_direct_rl.sh --episodes 1 --environment simple_grid

echo
echo "RL agent test completed!"