# OpenAI Codex Machine Learning Setup

## Introduction

The OpenAI Codex Machine Learning Setup is a comprehensive environment designed for building advanced AI-powered applications with a focus on agentic capabilities. This project provides a robust foundation for creating AI systems that can reason about complex tasks, interact with external tools, and execute actions on behalf of users.

Built around a core set of modern AI libraries and tools, this setup enables the development of sophisticated machine learning pipelines, particularly those leveraging Large Language Models (LLMs) for reasoning and decision-making. The project structure integrates seamlessly with FastMCP for standardized API interfaces and provides connectivity with various external services through tool integrations.

## Core Libraries

### AI & Orchestration

| Library | Version | Purpose |
|---------|---------|---------|
| **DSPy** | 2.6.24 | Declarative framework for building LLM-powered reasoning pipelines. DSPy enables structured programming of language model behavior, replacing brittle prompts with modular, maintainable code patterns. |
| **FastMCP** | 0.3.1 | Framework for creating standardized tool APIs via the Model Context Protocol (MCP). Serves as the interface layer between our AI systems and external clients or services. |
| **ArcadePy** | 1.4.0 | Python SDK for Arcade.dev, providing authenticated integrations to external services like GitHub, Slack, etc. Handles secure tool execution with proper credentialing. |
| **OpenAI** | 1.61.0 | Client library for interfacing with OpenAI's models. Used by DSPy for LLM-powered reasoning capabilities. |
| **LiteLLM** | 1.60.4 | Proxy routing service for LLMs, allowing DSPy to route calls to various model providers (OpenAI, Arcade, etc.). |

### Web Server & Routing

| Library | Version | Purpose |
|---------|---------|---------|
| **FastAPI** | 0.111.0 | Modern, high-performance web framework for building APIs. Used when exposing our agent as an HTTP service. |
| **Uvicorn** | 0.29.0 | ASGI server implementation, used to serve FastAPI applications with high performance. |

### HTTP & Utilities

| Library | Version | Purpose |
|---------|---------|---------|
| **Requests** | 2.32.3 | Standard HTTP client library for making direct API calls to external services. |
| **HTTPX** | 0.27.0 | Async-capable HTTP client that works well with FastAPI for non-blocking operations. |

### Auth & Security

| Library | Version | Purpose |
|---------|---------|---------|
| **python-dotenv** | 1.0.1 | Library for loading environment variables from a .env file, critical for securely managing API keys and credentials. |

### Testing & Quality Assurance

| Library | Version | Purpose |
|---------|---------|---------|
| **pytest** | 8.2.0 | Testing framework for writing and executing unit and integration tests. |
| **pytest-mock** | 3.14.0 | Extension for monkeypatching in tests, allowing controlled mocking of dependencies. |
| **pytest-cov** | 5.0.0 | Plugin for generating code coverage reports during test execution. |
| **Black** | 24.4.2 | Code formatter ensuring consistent styling across the codebase. |
| **Flake8** | 7.0.0 | Linter for enforcing coding standards and catching potential issues. |
| **MyPy** | 1.10.0 | Static type checker for Python, enhancing code quality and catching type-related errors. |

### Observability

| Library | Version | Purpose |
|---------|---------|---------|
| **Loguru** | 0.7.2 | Advanced logging library providing flexible and visually appealing logs. |
| **Rich** | 13.7.1 | Library for rich terminal output, enhancing the display of logs and debug information. |

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Virtual environment tool (recommended: venv or conda)

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-organization/codex-one.git
   cd codex-one
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the local MCP package for development**:
   ```bash
   pip install -e ./mcp
   ```

5. **Set up environment variables**:
   Create a `.env` file in the project root with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ARCADE_API_KEY=your_arcade_api_key
   # Add other required API keys/credentials
   ```

## Basic Usage

### Creating a Simple DSPy Pipeline

```python
import dspy
from dspy.retrieve import ColBERTv2

# Set up DSPy with OpenAI
llm = dspy.OpenAI(model="gpt-4o")
dspy.settings.configure(lm=llm)

# Define a simple QA module
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.generate_answer(question=question)

# Use the module
qa = SimpleQA()
response = qa(question="How do neural networks learn?")
print(response.answer)
```

### Setting Up a FastMCP Server

```python
from fastapi import FastAPI
from dspy_mcp.server.app import MCPServer
from dspy_mcp.tools.echo import EchoTool

# Create a FastAPI app
app = FastAPI(title="Codex AI Service")

# Initialize the MCP Server
mcp_server = MCPServer()

# Register a simple tool
mcp_server.register_tool(EchoTool())

# Mount the MCP server on the FastAPI app
app.mount("/mcp", mcp_server.app)

# Run with: uvicorn server:app --reload
```

### Using Arcade.dev Integration

```python
import arcadepy as arcade
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Arcade client
client = arcade.Client()

# Authenticate with Arcade
auth_info = client.auth.start(
    redirect_uri="http://localhost:3000/callback",
    scopes=["github", "slack"]
)

# After user authentication, use tools
github_tool = client.tools.github
issue = github_tool.create_issue(
    owner="your-org",
    repo="your-repo",
    title="New feature request",
    body="Implement AI-powered code review"
)

# Send notification via Slack
slack_tool = client.tools.slack
slack_tool.post_message(
    channel="#dev-team",
    text=f"Created GitHub issue: {issue.html_url}"
)
```

### Combining DSPy with Tools

```python
import dspy
from dspy_mcp.client.app import MCPClient

# Setup MCP client to connect to tools
mcp_client = MCPClient(base_url="http://localhost:8000/mcp")

# Register MCP tools with DSPy
tools = mcp_client.get_tools()
dspy.settings.configure(tools=tools)

# Create a ReAct agent that can use tools
agent = dspy.ReAct(tools=tools)

# Run the agent with a task
result = agent.forward(task="Create a GitHub issue about the login bug and notify the team on Slack")
print(result.answer)
```

## Next Steps

- Explore the `examples/` directory for more detailed usage scenarios
- Check the API documentation for comprehensive information about available modules and functions
- Join our community Discord for support and discussions
- Consider contributing to the project by submitting pull requests or raising issues

For more information on individual components:
- DSPy documentation: [https://dspy.ai](https://dspy.ai)
- Arcade.dev documentation: [https://docs.arcade.dev](https://docs.arcade.dev)
- FastMCP documentation: See `mcp/README.md`