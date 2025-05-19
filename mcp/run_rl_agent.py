#!/usr/bin/env python3
"""Example script to train and run the RL echo agent."""
import argparse
import os
import sys

# Ensure project root is on sys.path so local 'dspy' stub is importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dspy_mcp.pipeline.rl_pipeline import RLEchoAgent, train_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run the RL echo agent")
    parser.add_argument("--samples", type=int, default=5, help="Number of training samples")
    args = parser.parse_args()

    agent = RLEchoAgent()

    # Create a simple training dataset favoring 'upper' actions
    data = []
    for i in range(args.samples):
        action = "upper" if i % 2 == 0 else "lower"
        reward = 1.0 if action == "upper" else -1.0
        data.append((f"example{i}", action, reward))

    train_agent(agent, data)

    result = agent(text="hello")
    print(result.response)


if __name__ == "__main__":
    main()
