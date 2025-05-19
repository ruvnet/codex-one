# Using the Reinforcement Learning Agent in DSPy MCP

This document explains how to run and train the reinforcement learning agent implemented in the DSPy MCP server.

## Overview

The DSPy MCP server includes a simple reinforcement learning (RL) agent that learns to choose between uppercase and lowercase transformations for the echo tool. The agent uses DSPy's GRPO (Generalized Reweighted Policy Optimization) to learn a policy based on reward signals.

## Available Scripts

Two scripts are provided to help you run and test the components:

1. `run_dspy_agent.sh` - A bash script that sets up the environment and runs basic tests of the DSPy agent and MCP server
2. `run_rl_agent.py` - A Python script that runs and trains the RL agent

## Setting Up the Environment

Before running the scripts, you need to set up the environment:

```bash
# Navigate to the mcp directory
cd mcp

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .
```

## Running the Basic DSPy Agent

To run the basic DSPy agent and test the MCP server:

```bash
./run_dspy_agent.sh
```

This script will:
1. Set up the virtual environment
2. Install the package
3. Start the MCP server in SSE mode
4. Test both the `echo` and `dspy_echo` tools
5. Shut down the server

## Running the Reinforcement Learning Agent

To run and train the RL agent:

```bash
./run_rl_agent.py
```

By default, this script will:
1. Start the MCP server in SSE mode
2. Collect training data by calling the echo tool with different inputs
3. Train the RL agent using this data
4. Test the trained agent to see how its policy has changed
5. Shut down the server

### Command-Line Options

The RL agent script supports several command-line options:

```bash
./run_rl_agent.py --help
```

Available options:
- `--port PORT` - Specify the port for the MCP server (default: 3001)
- `--samples N` - Number of training samples to collect (default: 10)
- `--no-server` - Don't start a new server (use an existing one)

### Example Usage

```bash
# Run with 20 training samples
./run_rl_agent.py --samples 20

# Use an existing server
./run_rl_agent.py --no-server
```

## How the RL Agent Works

The RL agent (`RLEchoAgent`) learns to choose between uppercase and lowercase transformations based on rewards. In this example:

1. The agent starts with a 50% probability of choosing uppercase
2. Training data is collected with reward signals
3. The agent is trained to maximize reward
4. After training, the agent's probability of choosing uppercase is updated

The current implementation gives positive rewards for:
- Using uppercase for texts with exclamation marks
- Using lowercase for texts without exclamation marks

This is a simple example to demonstrate RL in DSPy - you can modify the reward function to implement more complex policies.

## Extending the RL Agent

To extend or modify the RL agent:

1. Edit `dspy_mcp/pipeline/rl_pipeline.py` to change the agent implementation
2. Modify `run_rl_agent.py` to change how training data is collected or how rewards are assigned
3. Run the updated script to see the effects of your changes

## Integration with MCP Server

The RL agent is not currently exposed directly through the MCP server API. To integrate it:

1. Edit `dspy_mcp/server/app.py` to add a new tool that uses the RL agent
2. Register the tool with the server in the `register_tools` function

Example:
```python
@mcp_server.tool(
    name="rl_echo",
    description="Echo text using the reinforcement learning agent",
)
def rl_echo_tool(text: str) -> types.TextContent:
    """Echo using the RL agent."""
    agent = RLEchoAgent()
    prediction = agent(text=text)
    return types.TextContent(type="text", text=prediction.response, format="text/plain")