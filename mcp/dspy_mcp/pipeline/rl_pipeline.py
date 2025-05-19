import random
from typing import List, Tuple

import dspy
from dspy_mcp.tools.echo import echo


class RLEchoAgent(dspy.Module):
    """DSPy-inspired agent that learns to echo with case transform."""

    def __init__(self):
        super().__init__()
        self.signature = dspy.Signature(
            {
                "text": (str, dspy.InputField()),
                "response": (str, dspy.OutputField()),
                "action": (str, dspy.OutputField()),
            }
        )
        self.prob_upper = dspy.Tunable("prob_upper", 0.5)

    def forward(self, text: str):
        use_upper = random.random() < self.prob_upper()
        action = "upper" if use_upper else "lower"
        result = echo(text, action)
        return dspy.Prediction(response=result.text, action=action)


def train_agent(agent: RLEchoAgent, data: List[Tuple[str, str, float]]) -> None:
    """Train the agent with (text, action, reward) tuples."""
    inputs = [(t, a) for t, a, _ in data]
    rewards = [r for _, _, r in data]
    optimizer = dspy.SimpleGRPO(agent, epochs=1)
    optimizer.train(inputs, rewards)
