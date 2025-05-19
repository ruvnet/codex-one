# DSPy Agent for MCP Server

This document explains how to use the DSPy agent implemented in this MCP server.

## Overview

The current implementation includes a simple DSPy-based echo agent that demonstrates the fundamental structure of an agent built with DSPy and exposed through the MCP (Model Context Protocol) server. This implementation serves as a starting point for developing more complex agents.

## Agent Architecture

The agent is structured as follows:

1. **Tool Implementation (`dspy_mcp/tools/echo.py`)**: Defines the core functionality that performs work
2. **Agent Pipeline (`dspy_mcp/pipeline/agent_pipeline.py`)**: Wraps the tool in a DSPy module
3. **Server Registration (`dspy_mcp/server/app.py`)**: Exposes the agent as an MCP tool

### The Echo Agent

The Echo Agent is a simple DSPy module that:

1. Takes text input and an optional transform parameter
2. Processes this input (in this case, just echoes it back, optionally transforming to upper/lower case)
3. Returns the response

The agent structure using DSPy:

```python
class EchoAgent(dspy.Module):
    """Simple DSPy module that delegates to the echo tool."""

    def __init__(self):
        super().__init__()
        self.signature = dspy.Signature(
            {
                "text": (str, dspy.InputField()),
                "transform": (Optional[str], dspy.InputField()),
                "response": (str, dspy.OutputField()),
            }
        )

    def forward(self, text: str, transform: Optional[str] = None):
        result = echo(text, transform)
        return dspy.Prediction(response=result.text)
```

## Running the DSPy Agent Directly

For testing purposes, you can run the DSPy agent directly using the provided script:

```bash
cd mcp
python simple_dspy_agent.py "Your text here" --transform upper
```

This will:
1. Call the echo tool directly with your text
2. Call the DSPy agent with the same text
3. Display both results for comparison

## Running the MCP Server with the Agent

The MCP server can be run in two different transport modes:

### SSE Transport (HTTP-based)

To run the MCP server with SSE transport (HTTP-based):

```bash
cd mcp
./run_mcp_server.sh
```

This starts an HTTP server on port 3001 that you can interact with using curl or other HTTP clients.

### stdio Transport (Terminal-based)

To run the MCP server with stdio transport (useful for integration with CLI tools):

```bash
cd mcp
./run_mcp_stdio.sh
```

This starts the MCP server in stdio mode, which reads from standard input and writes to standard output.

You can interact with the server using the provided client script:

```bash
# In one terminal, start the stdio server
./run_mcp_stdio.sh

# In another terminal, send requests to the server
echo '{"tool":"echo","params":{"text":"hello","transform":"upper"}}' | ./mcp_stdio_client.sh

# To use the DSPy agent
echo '{"tool":"dspy_echo","params":{"text":"testing dspy agent","transform":"lower"}}' | ./mcp_stdio_client.sh
```

You can interact with the agent through the MCP server API:

```bash
# Query the echo tool directly
curl -X POST http://localhost:3001/tools/echo \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello, World!", "transform": "upper"}'

# Query the DSPy agent
curl -X POST http://localhost:3001/tools/dspy_echo \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello, World!", "transform": "upper"}'
```

## Extending the Agent

To build more advanced agents:

1. Create new tool implementations in `dspy_mcp/tools/`
2. Develop DSPy modules in `dspy_mcp/pipeline/` that use these tools
3. Register the new modules as MCP tools in `dspy_mcp/server/app.py`

For example, to create a reinforcement learning agent:

1. Implement RL algorithms and environment in new tool modules
2. Create a DSPy module that wraps the RL logic
3. Register it as an MCP tool

## Running the Reinforcement Learning Agent

The RL agent can be run in multiple ways:

### Direct Execution (No MCP Server)

To run the RL agent directly without starting the MCP server:

```bash
cd mcp
./run_direct_rl.sh --environment simple_grid --episodes 10 --learning-rate 0.2
```

This script directly uses the DSPy RL agent module. Optional parameters include:
- `--environment`: The environment to use (default: simple_grid)
- `--episodes`: Number of episodes to run (default: 5)
- `--learning-rate`: Learning rate for the algorithm (default: 0.1)
- `--gamma`: Discount factor (default: 0.9)
- `--epsilon`: Exploration rate (default: 0.1)
- `--verbose`: Enable verbose output

### Through the MCP Server (stdio Transport)

Run the stdio server and use the test script:

```bash
# In one terminal, start the server
./run_mcp_stdio.sh

# In another terminal, test the RL agent
./test_rl_agent_stdio.sh
```

### Through the MCP Server (SSE Transport)

Run the SSE server and use curl:

```bash
# In one terminal, start the server
./run_mcp_server.sh

# In another terminal, call the RL agent
curl -X POST http://localhost:3001/tools/dspy_rl_agent \
  -H "Content-Type: application/json" \
  -d '{"environment":"simple_grid","episodes":5,"learning_rate":0.1,"gamma":0.9,"epsilon":0.1}'
```

## Next Steps

Future enhancements could include:

- Adding more sophisticated DSPy pipelines with chained modules
- Expanding the reinforcement learning capabilities with additional algorithms and environments
- Incorporating model routing for optimal performance
- Adding data persistence to learn from previous interactions