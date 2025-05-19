#!/usr/bin/env python3
"""
Direct reinforcement learning agent runner for the DSPy MCP project.
This script allows running the RL agent directly without going through the MCP server.
"""

import argparse
import sys
import json
from typing import Optional, Dict, Any

# Add the current directory to the path so we can import the DSPy MCP package
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the DSPy RL agent
from dspy_mcp.pipeline.rl_agent import RLAgent

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the DSPy RL agent directly')
    parser.add_argument('--environment', type=str, default='simple_grid',
                        help='The environment to run the RL agent in')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate for the RL algorithm')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Exploration rate')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()

def main():
    """Main function to run the RL agent."""
    args = parse_args()
    
    try:
        # Create the RL agent
        agent = RLAgent()
        
        # Run the agent
        print(f"Running RL agent with environment={args.environment}, episodes={args.episodes}")
        print(f"Parameters: learning_rate={args.learning_rate}, gamma={args.gamma}, epsilon={args.epsilon}")
        
        # Call the run method with the specified parameters
        result = agent.forward(
            environment=args.environment,
            episodes=args.episodes,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon=args.epsilon
        )
        
        # Print the result
        print("\nRL Agent Result:")
        if hasattr(result, 'result'):
            print(result.result)
        else:
            print(result)
            
    except Exception as e:
        print(f"Error running RL agent: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()