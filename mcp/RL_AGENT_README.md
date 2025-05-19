# Reinforcement Learning Agent for DSPy MCP

This document provides detailed information about the reinforcement learning (RL) agent implementation for the DSPy MCP server.

## Overview

The RL agent implements a Q-learning algorithm to solve various environments, with the simple grid world environment as the default example. The agent is built using the DSPy framework and can be accessed either directly through Python or via the MCP server interface.

## Algorithm: Q-Learning

The agent implements Q-learning, a model-free reinforcement learning algorithm that learns the value of an action in a particular state. The algorithm follows these steps:

1. Initialize Q-values table with zeros
2. For each episode:
   - Choose a starting state
   - Until the goal state is reached or max steps exceeded:
     - Choose an action (using epsilon-greedy strategy)
     - Take the action, observe reward and next state
     - Update Q-value using the formula:
       Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
     - Move to the next state

### Parameters

The algorithm can be configured with the following parameters:

- `learning_rate` (α): How much the agent values new information over existing information
- `gamma` (γ): Discount factor that determines the importance of future rewards
- `epsilon` (ε): The probability of choosing a random action (exploration)
- `episodes`: Number of episodes to run during training

## Environments

### Simple Grid World

The default environment is a simple grid world where the agent must navigate from a starting position to a goal position. The grid contains:

- Empty cells that the agent can move through
- Obstacles that the agent cannot pass
- A goal cell that provides a positive reward
- Optional trap cells that provide negative rewards

The agent can take four actions: up, down, left, and right. If the agent attempts to move into an obstacle or out of bounds, it remains in its current position.

### Custom Environments

Additional environments can be implemented by extending the base `Environment` class and registering them with the RL agent.

## Integration with DSPy

The RL agent is implemented as a DSPy module:

```python
class RLAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = dspy.Signature(
            {
                "environment": (str, dspy.InputField()),
                "episodes": (int, dspy.InputField(default=10)),
                "learning_rate": (float, dspy.InputField(default=0.1)),
                "gamma": (float, dspy.InputField(default=0.9)),
                "epsilon": (float, dspy.InputField(default=0.1)),
                "result": (str, dspy.OutputField()),
            }
        )
        # Initialize environments, Q-table, etc.

    def forward(self, environment, episodes=10, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        # Run the reinforcement learning algorithm
        # Return the results
        return dspy.Prediction(result="Training completed: " + stats_summary)
```

## Running the RL Agent

### Direct Execution

```bash
./run_direct_rl.sh --environment simple_grid --episodes 10 --learning-rate 0.2
```

### Via MCP Server (stdio)

```bash
# Start the server in stdio mode
./run_mcp_stdio.sh

# In another terminal
./test_rl_agent_stdio.sh
```

### Via MCP Server (SSE)

```bash
# Start the server with HTTP interface
./run_mcp_server.sh

# In another terminal
curl -X POST http://localhost:3001/tools/dspy_rl_agent \
  -H "Content-Type: application/json" \
  -d '{"environment":"simple_grid","episodes":5,"learning_rate":0.1,"gamma":0.9,"epsilon":0.1}'
```

## Implementation Details

The RL agent implementation consists of several components:

1. **Environment Interface**: Defines the interaction between the agent and environment
2. **Q-Learning Algorithm**: Implements the core learning algorithm
3. **Policy**: Defines how actions are selected (e.g., epsilon-greedy)
4. **Experience Memory**: Stores and replays past experiences (for more advanced implementations)

## Expected Output

The RL agent returns a formatted string containing:
- Training statistics (average reward, success rate)
- Final Q-values for important states
- Visualization of the learned policy

Example:
```
Training completed after 100 episodes
Average reward: 8.5
Success rate: 92%
Optimal path found: Start → (0,1) → (1,1) → (2,1) → (2,2) → Goal
```

## Extending the RL Agent

To add new reinforcement learning algorithms or environments:

1. Implement new environment classes in the environments directory
2. Create new algorithm classes (e.g., for SARSA or DQN) in the algorithms directory
3. Register them with the RL agent module
4. Update the DSPy module's forward method to handle the new options

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
- DSPy documentation: https://dspy-ai.github.io/
- MCP Server documentation