import argparse
import os
import sys

# Ensure imports work when script run from this folder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dspy_mcp.pipeline.rl_pipeline import RLEchoAgent, train_agent


def generate_dataset(num_samples: int):
    data = []
    for i in range(num_samples):
        text = f"sample{i}"
        action = "upper" if i % 2 == 0 else "lower"
        reward = 1.0 if action == "upper" else -1.0
        data.append((text, action, reward))
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and run the RL echo agent")
    parser.add_argument("--samples", type=int, default=4, help="number of training samples")
    args = parser.parse_args(argv)

    agent = RLEchoAgent()
    dataset = generate_dataset(args.samples)
    train_agent(agent, dataset)

    print(f"Trained prob_upper: {agent.prob_upper()}")
    for text, _action, _reward in dataset:
        result = agent(text=text)
        print(f"{text} -> {result.response} ({result.action})")


if __name__ == "__main__":
    main()
