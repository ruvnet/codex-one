"""
Reinforcement Learning Agent implementation for the DSPy MCP server.
This module implements a Q-learning based agent for solving grid world environments.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import dspy
from enum import Enum

class Action(Enum):
    """Possible actions in the grid world environment."""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class CellType(Enum):
    """Types of cells in the grid world environment."""
    EMPTY = 0
    OBSTACLE = 1
    GOAL = 2
    TRAP = 3
    START = 4

class SimpleGridEnvironment:
    """A simple grid world environment for reinforcement learning."""
    
    def __init__(self, grid_size: Tuple[int, int] = (5, 5), 
                 start_pos: Tuple[int, int] = (0, 0),
                 goal_pos: Tuple[int, int] = (4, 4),
                 obstacles: List[Tuple[int, int]] = None,
                 traps: List[Tuple[int, int]] = None,
                 max_steps: int = 100):
        """
        Initialize the grid world environment.
        
        Args:
            grid_size: Tuple of (height, width)
            start_pos: Starting position (row, col)
            goal_pos: Goal position (row, col)
            obstacles: List of obstacle positions (row, col)
            traps: List of trap positions (row, col)
            max_steps: Maximum steps per episode
        """
        self.height, self.width = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = obstacles or []
        self.traps = traps or []
        self.max_steps = max_steps
        
        # Initialize the grid
        self.grid = self._create_grid()
        
        # Current state
        self.current_pos = start_pos
        self.step_count = 0
        self.done = False
    
    def _create_grid(self) -> List[List[CellType]]:
        """Create the grid with obstacles, goals, and traps."""
        grid = [[CellType.EMPTY for _ in range(self.width)] for _ in range(self.height)]
        
        # Place obstacles
        for obstacle in self.obstacles:
            grid[obstacle[0]][obstacle[1]] = CellType.OBSTACLE
        
        # Place traps
        for trap in self.traps:
            grid[trap[0]][trap[1]] = CellType.TRAP
        
        # Place goal
        grid[self.goal_pos[0]][self.goal_pos[1]] = CellType.GOAL
        
        # Place start
        grid[self.start_pos[0]][self.start_pos[1]] = CellType.START
        
        return grid
    
    def reset(self) -> Tuple[int, int]:
        """Reset the environment to initial state."""
        self.current_pos = self.start_pos
        self.step_count = 0
        self.done = False
        return self.current_pos
    
    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.step_count += 1
        
        # Calculate next position
        row, col = self.current_pos
        if action == Action.UP:
            next_pos = (max(0, row - 1), col)
        elif action == Action.RIGHT:
            next_pos = (row, min(self.width - 1, col + 1))
        elif action == Action.DOWN:
            next_pos = (min(self.height - 1, row + 1), col)
        elif action == Action.LEFT:
            next_pos = (row, max(0, col - 1))
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Check if next position is an obstacle
        next_row, next_col = next_pos
        if self.grid[next_row][next_col] == CellType.OBSTACLE:
            next_pos = self.current_pos  # Stay in current position
        
        # Update current position
        self.current_pos = next_pos
        
        # Calculate reward
        reward = -0.1  # Small negative reward for each step (encourages efficiency)
        
        # Check if reached goal
        if self.grid[next_row][next_col] == CellType.GOAL:
            reward = 10.0
            self.done = True
        
        # Check if stepped on a trap
        elif self.grid[next_row][next_col] == CellType.TRAP:
            reward = -5.0
        
        # Check if maximum steps reached
        if self.step_count >= self.max_steps:
            self.done = True
        
        info = {
            "step_count": self.step_count,
            "max_steps_reached": self.step_count >= self.max_steps
        }
        
        return next_pos, reward, self.done, info
    
    def get_state_space_size(self) -> int:
        """Get the size of the state space."""
        return self.height * self.width
    
    def get_action_space_size(self) -> int:
        """Get the size of the action space."""
        return len(Action)
    
    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert state tuple to a flat index."""
        row, col = state
        return row * self.width + col
    
    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert flat index to state tuple."""
        row = index // self.width
        col = index % self.width
        return (row, col)
    
    def render(self) -> str:
        """Render the grid as a string."""
        output = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                if (i, j) == self.current_pos:
                    row.append("A")  # Agent
                elif self.grid[i][j] == CellType.OBSTACLE:
                    row.append("#")  # Obstacle
                elif self.grid[i][j] == CellType.GOAL:
                    row.append("G")  # Goal
                elif self.grid[i][j] == CellType.TRAP:
                    row.append("T")  # Trap
                elif self.grid[i][j] == CellType.START:
                    row.append("S")  # Start
                else:
                    row.append(".")  # Empty
            output.append("".join(row))
        return "\n".join(output)


class QLearningAgent:
    """
    Q-learning agent for reinforcement learning.
    """
    
    def __init__(self, environment, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        """
        Initialize the Q-learning agent.
        
        Args:
            environment: The environment to interact with
            learning_rate: Learning rate (alpha)
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.env = environment
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table with zeros
        state_space_size = self.env.get_state_space_size()
        action_space_size = self.env.get_action_space_size()
        self.q_table = np.zeros((state_space_size, action_space_size))
        
        # Statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_success = []
    
    def choose_action(self, state: Tuple[int, int]) -> Action:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Action to take
        """
        state_idx = self.env.state_to_index(state)
        
        # Exploration: choose a random action
        if random.random() < self.epsilon:
            return Action(random.randint(0, len(Action) - 1))
        
        # Exploitation: choose the best action
        return Action(np.argmax(self.q_table[state_idx]))
    
    def update_q_table(self, state: Tuple[int, int], action: Action, 
                       reward: float, next_state: Tuple[int, int]) -> None:
        """
        Update the Q-table using the Q-learning formula.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        state_idx = self.env.state_to_index(state)
        next_state_idx = self.env.state_to_index(next_state)
        
        # Q-learning formula: Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        old_value = self.q_table[state_idx][action.value]
        next_max = np.max(self.q_table[next_state_idx])
        
        new_value = old_value + self.learning_rate * (reward + self.gamma * next_max - old_value)
        self.q_table[state_idx][action.value] = new_value
    
    def train(self, episodes: int) -> Dict[str, Any]:
        """
        Train the agent for a specified number of episodes.
        
        Args:
            episodes: Number of episodes to train
            
        Returns:
            Dictionary with training statistics
        """
        for episode in range(episodes):
            # Reset environment
            state = self.env.reset()
            total_reward = 0
            step_count = 0
            done = False
            
            # Episode loop
            while not done:
                # Choose and take action
                action = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Update Q-table
                self.update_q_table(state, action, reward, next_state)
                
                # Update statistics
                total_reward += reward
                step_count += 1
                
                # Update state
                state = next_state
            
            # Record episode statistics
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(step_count)
            self.episode_success.append(info.get("step_count", 0) < self.env.max_steps and done)
        
        # Return training statistics
        return {
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
            "episode_success": self.episode_success,
            "average_reward": sum(self.episode_rewards) / episodes,
            "success_rate": sum(self.episode_success) / episodes * 100,
            "q_table": self.q_table
        }
    
    def get_optimal_policy(self) -> List[Action]:
        """Get the optimal policy from the Q-table."""
        policy = []
        for i in range(self.env.get_state_space_size()):
            state = self.env.index_to_state(i)
            state_idx = self.env.state_to_index(state)
            best_action = Action(np.argmax(self.q_table[state_idx]))
            policy.append(best_action)
        return policy
    
    def visualize_policy(self) -> str:
        """Visualize the optimal policy."""
        policy = self.get_optimal_policy()
        output = []
        
        for i in range(self.env.height):
            row = []
            for j in range(self.env.width):
                state_idx = self.env.state_to_index((i, j))
                
                if (i, j) == self.env.goal_pos:
                    row.append("G")  # Goal
                elif (i, j) in self.env.obstacles:
                    row.append("#")  # Obstacle
                elif (i, j) in self.env.traps:
                    row.append("T")  # Trap
                else:
                    action = policy[state_idx]
                    if action == Action.UP:
                        row.append("↑")
                    elif action == Action.RIGHT:
                        row.append("→")
                    elif action == Action.DOWN:
                        row.append("↓")
                    elif action == Action.LEFT:
                        row.append("←")
            
            output.append("".join(row))
        
        return "\n".join(output)


def create_environment(env_name: str) -> Any:
    """
    Create an environment based on the name.
    
    Args:
        env_name: Name of the environment
        
    Returns:
        Environment instance
    """
    if env_name == "simple_grid":
        # Create a simple grid world
        obstacles = [(1, 1), (1, 3), (3, 1), (3, 3)]
        traps = [(2, 3)]
        return SimpleGridEnvironment(
            grid_size=(5, 5),
            start_pos=(0, 0),
            goal_pos=(4, 4),
            obstacles=obstacles,
            traps=traps,
            max_steps=50
        )
    elif env_name == "simple_grid_large":
        # Create a larger grid world
        obstacles = [(1, 1), (1, 3), (1, 5), (3, 1), (3, 3), (3, 5), (5, 1), (5, 3), (5, 5)]
        traps = [(2, 3), (4, 5)]
        return SimpleGridEnvironment(
            grid_size=(7, 7),
            start_pos=(0, 0),
            goal_pos=(6, 6),
            obstacles=obstacles,
            traps=traps,
            max_steps=100
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")


class RLAgent(dspy.Module):
    """
    DSPy module for reinforcement learning agent.
    """
    
    def __init__(self):
        """Initialize the RL agent module."""
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
    
    def forward(self, environment: str, episodes: int = 10, 
                learning_rate: float = 0.1, gamma: float = 0.9, 
                epsilon: float = 0.1) -> dspy.Prediction:
        """
        Run the reinforcement learning algorithm.
        
        Args:
            environment: Name of the environment
            episodes: Number of episodes to train
            learning_rate: Learning rate (alpha)
            gamma: Discount factor
            epsilon: Exploration rate
            
        Returns:
            DSPy prediction with result string
        """
        # Create environment
        try:
            env = create_environment(environment)
        except ValueError as e:
            return dspy.Prediction(result=f"Error: {str(e)}")
        
        # Create agent
        agent = QLearningAgent(env, learning_rate=learning_rate, gamma=gamma, epsilon=epsilon)
        
        # Train agent
        print(f"Training RL agent for {episodes} episodes...")
        stats = agent.train(episodes)
        
        # Generate result output
        output = []
        output.append(f"Training completed after {episodes} episodes")
        output.append(f"Average reward: {stats['average_reward']:.2f}")
        output.append(f"Success rate: {stats['success_rate']:.1f}%")
        output.append("\nLearned Policy:")
        output.append(agent.visualize_policy())
        
        return dspy.Prediction(result="\n".join(output))