from dspy_mcp.pipeline.agent_pipeline import run_agent
from mcp import types
import dspy_mcp.pipeline.agent_pipeline as pipeline


def test_run_agent_calls_echo(monkeypatch):
    called = {}

    def fake_echo(text: str, transform=None):
        called['called'] = text
        return types.TextContent(type="text", text=text[::-1], format="text/plain")

    monkeypatch.setattr(pipeline, 'echo', fake_echo)

    result = run_agent("hello")
    assert called['called'] == "hello"
    assert result == "olleh"
