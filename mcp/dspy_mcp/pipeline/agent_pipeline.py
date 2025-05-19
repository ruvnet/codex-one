import dspy
from typing import Optional

from dspy_mcp.tools.echo import echo

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


echo_agent = EchoAgent()

def run_agent(text: str, transform: Optional[str] = None) -> str:
    """Run the EchoAgent and return the response string."""
    prediction = echo_agent(text=text, transform=transform)
    return prediction.response
