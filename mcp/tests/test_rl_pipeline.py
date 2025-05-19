import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dspy_mcp.pipeline.rl_pipeline import RLEchoAgent, train_agent


class TestRLPipeline(unittest.TestCase):
    def test_training_updates_policy(self):
        agent = RLEchoAgent()
        data = [
            ("hi", "upper", 1.0),
            ("hi", "lower", -1.0),
        ]
        train_agent(agent, data)
        self.assertGreater(agent.prob_upper(), 0.5)


if __name__ == '__main__':
    unittest.main()
