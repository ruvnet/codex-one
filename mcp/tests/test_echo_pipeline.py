import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dspy_mcp.pipeline.agent_pipeline import run_agent
import dspy_mcp.pipeline.agent_pipeline as pipeline
from mcp import types


class TestEchoPipeline(unittest.TestCase):
    def test_run_agent_calls_echo(self):
        called = {}

        def fake_echo(text: str, transform=None):
            called['called'] = text
            return types.TextContent(type="text", text=text[::-1], format="text/plain")

        with mock.patch.object(pipeline, 'echo', fake_echo):
            result = run_agent("hello")

        self.assertEqual(called['called'], "hello")
        self.assertEqual(result, "olleh")


if __name__ == '__main__':
    unittest.main()
