
# dspy-mcp

Great. I’ll build out a complete implementation spec and file structure for a Switchpoint-style LLM router optimized for coding intent, using DSPY for model routing and reinforcement learning, LiteLLM Proxy for local inference, and cloud APIs including OpenAI, Anthropic, and Google Gemini. It will include:

* Optical intent-based routing for code generation or error correction
* FastAPI gateway (fastmcp API) for completions and streaming
* LiteLLM proxy serving local models via Ollama
* A lightweight RL loop using DSPY to evolve routing logic

I’ll share the full directory structure, config files, API implementation, and logic shortly.


# Design and Implementation of a Code-Focused LLM Routing System (Switchpoint.dev Clone)

## Introduction and Goals

Switchpoint.dev demonstrated an **intelligent LLM routing** system that acts as a drop-in replacement for the OpenAI API, but dynamically chooses between multiple models to optimize cost and performance. Our goal is to build a **full clone of Switchpoint.dev** specialized for coding-related requests. This system will route user prompts about code (code completions, error debugging, refactoring suggestions, etc.) to the most suitable large language model based on the **intent of the request**, complexity, and past performance. We will incorporate the following key technologies:

* **LiteLLM Proxy + Ollama** for serving local open-source code models behind an OpenAI-compatible API (to reduce cost by handling simple cases locally).
* **Cloud LLM APIs** (OpenAI GPT-4, Anthropic Claude, Google Gemini, etc.) for higher-capability models when needed.
* **DSPy (Declarative Self-Improving Python)** for implementing the routing logic as a modular program and applying reinforcement learning to continuously improve model selection.
* **FastAPI Gateway** providing an OpenAI-compatible `/v1/chat/completions` endpoint with streaming support, so clients can use this service just like they use the OpenAI API.

By combining these, the system will automatically route coding queries (code generation vs error explanation vs refactoring requests) to the best model for the job, using cheaper/faster models by default and escalating to more powerful ones only when needed. Over time, a reinforcement learning loop will tune the routing policy based on feedback and usage logs, improving cost-effectiveness and quality.

## Architecture Overview

The architecture consists of modular components working together in a pipeline:

&#x20;*Figure: A conceptual architecture where a central LiteLLM-based router connects the gateway to multiple model backends (OpenAI GPT-4, Anthropic Claude, Google Gemini, local models via Ollama). The gateway receives client requests and delegates to the router, which then calls the appropriate model API.*

* **FastAPI Gateway** – The entry point exposing a chat completion API. It receives the user's request (chat messages and parameters) and streams back the completion. The gateway forwards requests to the routing logic module.
* **Routing Logic (DSPy)** – The “brain” of the system implemented with DSPy modules. It analyzes the request to determine the *intent* (e.g. code completion vs error explanation vs refactoring) and the *complexity/context size*. Based on this, it selects a target model or sequence of models. Initially, this can use heuristic rules (and small classifiers) to map intent to an optimal model choice. Over time, DSPy’s RL optimizers will adjust this routing policy for better outcomes.
* **Local Model Proxy (LiteLLM + Ollama)** – A service that hosts local code-oriented LLMs (like CodeLlama or WizardCoder) via an OpenAI-compatible API. The router can send requests here for low-cost inference on simpler tasks. Ollama is used under the hood to efficiently run these models on local hardware.
* **Cloud Model APIs** – Integrations to external APIs: OpenAI (GPT-4, GPT-3.5, etc.), Anthropic (Claude variants), and Google’s Gemini (next-gen PaLM) are configured. The router can call these via their SDKs or via the LiteLLM proxy (since LiteLLM supports many providers through a unified interface).
* **Reinforcement Learning Trainer (DSPy + GRPO)** – An offline component that continually learns from logs and feedback. It uses DSPy’s training loop (e.g. the GRPO algorithm) to adjust the routing strategy. Essentially, it treats model selection as a decision policy and tunes it to maximize reward (success rate, solution quality) minus costs (latency, token usage).

The system is designed to be **extensible** – new models can be added (update LiteLLM config or API keys), and new intents or routing rules can be learned via the RL mechanism. All components run in Docker containers orchestrated by **Docker Compose**, making it easy to deploy locally or on a server cluster.

## API Gateway (FastAPI Implementation)

The API gateway is a FastAPI application mimicking the OpenAI Chat Completions API. This allows existing developer tools or SDKs to use our service without code changes (just point the API base URL to our gateway). Key features of the gateway:

* **HTTP Endpoint**: `POST /v1/chat/completions` accepting JSON similar to OpenAI (e.g. `{"model": "...", "messages": [...], "stream": true}`).
* **Streaming Support**: If `stream=True`, the gateway will **stream partial results** as they are generated, in compatible chunked format (similar to OpenAI’s SSE). This is done using FastAPI’s `StreamingResponse` or by iterating an async generator that yields model output tokens.
* **Request Handling**: Upon receiving a request, the gateway wraps the user’s message(s) into a DSPy routing call. It calls the router module (explained below) which returns a final response or a generator of tokens.

**Gateway Code Sample (FastAPI + Uvicorn)**:

```python
# gateway/main.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from router.router import route_request  # our DSPy router function/module
import asyncio

app = FastAPI()

@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    # (Optional) You can parse model name if provided, to allow users to force a specific model.
    user_selected_model = body.get("model", None)
    
    # Call the routing logic to get an async generator or response
    result = route_request(messages, user_selected_model=user_selected_model, stream=stream)
    
    if stream:
        # Ensure result is an async generator of text chunks
        async def token_stream():
            async for token in result:
                # Format each token as an SSE-like or chunked response
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(token_stream(), media_type="text/event-stream")
    else:
        # Non-streaming: gather the full completion
        full_response = await (result if hasattr(result, "__await__") else asyncio.to_thread(lambda: result))
        return {"id": "<gen-id>", "object": "chat.completion", "model": full_response.model, 
                "choices": [{"message": full_response.message, "finish_reason": "stop", "index": 0}], 
                "usage": full_response.usage}
```

In the snippet above, `route_request` is the entry to our routing logic (could be a DSPy module or a wrapper function). If streaming is requested, we iterate over the tokens from `route_request` and yield them as server-sent events. Otherwise, we return the fully assembled completion. The gateway is intentionally simple – it mainly delegates the heavy logic to the router and ensures compatibility with OpenAI’s API format for requests and responses.

*Streaming:* The gateway uses `StreamingResponse` with an async generator to push tokens as they come. Each chunk is prefixed with `data: ` as per SSE convention, and `data: [DONE]` is sent at the end. This approach allows the client to start receiving the answer with low latency, even if the router had to try multiple models.

**Notes:** We include the optional ability for a user to specify a `model` in the request. In a true drop-in scenario, clients might send a model name (like `"gpt-4"`). Our router can interpret that as a *hint or override* – for example, if `model == "gpt-4"` we might bypass routing logic and call GPT-4 directly (or map it to an equivalent). Otherwise, we ignore the model field and perform automatic routing.

## Routing Logic with DSPy (Intent Detection and Model Selection)

The router is the core of the system: it decides **which model(s)** to use for a given request. We implement this using DSPy to allow easy experimentation and reinforcement learning on the routing policy. The logic can be broken down into steps:

1. **Intent Classification** – Determine the type of coding query:

   * *Completion*: The user wants code to be written or completed (e.g. “Write a Python function that does X”).
   * *Error Analysis*: The user provides an error or stack trace and asks for an explanation or fix (e.g. “Here’s an error I got, what does it mean?”).
   * *Refactoring*: The user gives some code and asks to improve or refactor it (e.g. “Refactor the following function for efficiency.”).

   This can be done via lightweight analysis of the prompt (e.g. presence of words like “error”/“exception” -> error intent; presence of code blocks and words like “refactor” -> refactor intent; otherwise default to completion). We can also leverage a small language model or classifier for intent detection if needed. For example, a DSPy **Module** could be defined to classify the prompt into one of these intents using a few-shot prompt or a fine-tuned model. In Switchpoint’s original system, they employed small fine-tuned models to identify the subject matter and difficulty of the task. We could similarly fine-tune a lightweight classifier on coding prompts or use rule-based heuristics initially.

2. **Model Mapping Strategy** – Based on the intent (and other factors like prompt length, complexity, past performance), choose a model or sequence. The strategy might be:

   * **Code Completion**: Try a fast, cost-effective code model first. For example, use a local CodeLlama-7B via Ollama for small tasks. If the prompt is large or the local model’s output confidence is low, escalate to a stronger model (like OpenAI GPT-4 or Claude). We maintain an ordered list of model options for this intent, along with conditions for upgrading. For instance, start with `wizardcoder-7B` (local) and fall back to `GPT-4` if the local result seems insufficient.
   * **Error Explanation**: These often require deeper reasoning and understanding of error messages. We might directly use a more capable model with a longer context window for reliability. For example, route error-related queries to **Anthropic Claude Instant** (which has strong reasoning and 100k context) or GPT-4-32k if the error log is very long. If cost is a concern, an intermediate model like **GPT-3.5 Turbo** can be tried for simpler errors, upgrading to Claude/GPT-4 only if the answer is not satisfactory (determined via feedback or a retry prompt).
   * **Code Refactoring**: If the provided code is short, a smaller model (or even an AI code tool) might suffice. But for larger code or complex refactoring (e.g., optimizing algorithms), a more advanced model like GPT-4 or Gemini might be needed. We can set a threshold on code length: e.g. if code > N lines, use a model with extended context (Claude or Gemini). Otherwise, attempt with a mid-tier model first (perhaps Google’s code model from Gemini or PaLM family, known to be strong in code).
   * **General fallback**: If an intent is unclear or the initial model fails to produce any answer, the router can escalate to the most powerful available model (GPT-4) as a safety net.

   The mapping strategy is essentially a decision tree or policy. For instance:

   * **If** intent is "error" **then** use Claude (fast, large context) -> if Claude fails or is not confident, try GPT-4.
   * **If** intent is "refactor" **then** if code length < 100 lines use CodeLlama 13B, else use GPT-4-32k.
   * **If** intent is "completion" **then** try local 7B -> if output quality low or user asks for more, use GPT-4.

   These rules combine *latency* (we prefer faster models like local or Anthropic Instant for quick response), *cost* (prefer free/cheap models first), and *expected performance* (complex tasks go to models known to excel in that area). We base the choices on public benchmarks and our own testing of these models on coding tasks. Over time, the RL loop will adjust the weights or thresholds in these rules.

3. **Execution and Streaming** – The router executes the chosen model call. Thanks to LiteLLM, this can be done through a unified interface. If the chosen model supports streaming, we stream the output tokens back to the gateway. If the router is using a fallback strategy (like try one model, then another on failure), it might need to buffer or detect failure conditions. For example, the router could start streaming from the first model, but if it detects a certain trigger (like the model outputs "I don't know" or some confidence score is low), it can switch to a different model mid-response. (Switchpoint mentions *“streams responses and upgrades on failure”*; implementing seamless mid-stream upgrade is complex, but one approach is to quickly invoke the better model and continue streaming its output).

**Router Implementation (DSPy pseudo-code)**:

We can implement the above logic as a DSPy program consisting of sub-modules for each intent and a top-level router module. For clarity, here’s a simplified pseudo-code using Python logic (DSPy would allow a more declarative approach, but this illustrates the flow):

```python
# router/router.py
import dspy
from litellm import openai_completion  # LiteLLM SDK or use requests

# Define functions or DSPy Modules for each type of request:
def handle_completion(messages, stream=True):
    # Try local model via LiteLLM first
    try:
        return call_model_via_litellm("local-code-model", messages, stream=stream)
    except Exception as e:
        # Fallback to GPT-4 if local fails
        return call_model_via_litellm("gpt-4", messages, stream=stream)

def handle_error_analysis(messages, stream=True):
    # Use Claude Instant (fast, good at reasoning) first
    try:
        return call_model_via_litellm("claude-instant", messages, stream=stream)
    except Exception:
        return call_model_via_litellm("gpt-4", messages, stream=stream)

def handle_refactoring(messages, stream=True):
    # Decide based on code length
    code_content = extract_code_from_messages(messages)
    if code_content and len(code_content.splitlines()) > 100:
        # Large code - use a model with extended context (Claude or GPT-4-32k)
        model_name = "claude-100k"
    else:
        # Smaller code - try a mid-tier model (Gemini or 3.5-turbo)
        model_name = "gemini-code"  # assume a Gemini code model is configured
    return call_model_via_litellm(model_name, messages, stream=stream)

def route_request(messages, user_selected_model=None, stream=True):
    """Main router function: determines intent and routes to appropriate handler."""
    user_query = messages[-1]['content'] if messages else ""  # last user message
    intent = classify_intent(user_query)  # e.g., "completion", "error", "refactor"
    if user_selected_model:
        # If user explicitly requested a model, bypass intent routing (direct call)
        return call_model_via_litellm(user_selected_model, messages, stream=stream)
    if intent == "error":
        return handle_error_analysis(messages, stream=stream)
    elif intent == "refactor":
        return handle_refactoring(messages, stream=stream)
    else:
        # Default to completion 
        return handle_completion(messages, stream=stream)
```

In this code:

* `classify_intent` would implement the rules or use a small model to return "completion"/"error"/"refactor".
* `call_model_via_litellm(model_name, messages, stream)` would invoke the model through the LiteLLM proxy. We could use the LiteLLM Python SDK or simply send an HTTP request to the LiteLLM server’s OpenAI-like endpoint (e.g., `POST http://litellm:4000/v1/chat/completions` with `{"model": model_name, "messages": [...]}`).

**Example**: If `messages[-1]['content']` contains the phrase "Traceback" or "Error:", `classify_intent` returns `"error"`. The router then calls `handle_error_analysis`, which by default attempts Claude. If Claude’s API fails or returns an error (or perhaps if the answer seems insufficient by some heuristic), it falls back to GPT-4. In either case, the final answer (or stream of tokens) is returned up to the FastAPI gateway to send back to the user.

We also log each request’s details to a **usage log** (could be a database or just JSON lines file). The log would include: the prompt, chosen intent, chosen model, and possibly outcome (success/failure, tokens used, latency, and any user feedback).

**DSPy integration**: The above is a straightforward Python logic. With DSPy, we could formalize this as a **multi-module graph**:

* A DSPy `Module` for intent classification (could be a Prompt-based classifier or even a `Tool` that runs a Python function to detect keywords).
* Modules for each action (one that calls local model, one that calls GPT-4, etc., perhaps using `dspy.LM` wrappers for each provider).
* A top-level module that *selects* which sub-module to execute based on the classifier output. DSPy might allow this via a control flow or by weighting different modules and having the optimizer tune those weights.

For example, one could create a DSPy `Signature` that includes the possible responses from each model and then use a *mixture-of-experts* approach with weights that the RL algorithm can adjust. Initially, those weights implement the heuristics above (like weight = 1 for Claude on error queries, etc.), and the RL training will refine them.

## Local Model Serving with LiteLLM and Ollama

To minimize cost, our system uses local **open-source LLMs** for many coding tasks. We leverage **LiteLLM** as a local proxy server that can host models via an OpenAI-compatible API. LiteLLM supports running models with backends like Ollama, which is optimized for running LLaMA-family models on local hardware (including Mac M-series or Linux with GPUs).

**Why LiteLLM?** It provides a unified interface to local and remote models. We can register our local code model (say, *CodeLlama 13B* or *WizardCoder 15B*) in a configuration, and also include entries for the cloud models with their API keys. The FastAPI router can then call **one service (LiteLLM)** regardless of which model is needed, simply by specifying the model name. LiteLLM will route the request to either the local inference (via Ollama) or out to OpenAI/Anthropic APIs as configured. This simplifies implementation since we don’t have to individually call different APIs; LiteLLM abstracts that away.

**LiteLLM Configuration (example `litellm_config.yaml`):**

```yaml
model_list:
  - model_name: local-code-model        # Alias we use in code
    litellm_params:
      model: ollama/CodeLlama-13B-Python       # Model served via Ollama
      # (We assume Ollama is running and has this model pulled. LiteLLM will forward to it.)
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4                # OpenAI GPT-4 (uses OPENAI_API_KEY from env)
  - model_name: claude-instant
    litellm_params:
      model: anthropic/claude-2         # Anthropic Claude Instant
  - model_name: claude-100k
    litellm_params:
      model: anthropic/claude-2-100k    # (if available, a Claude version with 100k context)
  - model_name: gemini-code
    litellm_params:
      model: gemini/gemini-1.5-pro-latest  # Google Gemini model via Vertex AI or unified API
  # ... you can add more models or variants as needed
```

In this YAML, we define model aliases and how LiteLLM should fulfill them. For local models, `ollama/ModelName` indicates to use the Ollama backend (which needs the model installed via Ollama CLI). For remote models, we use the provider prefix (`openai/`, `anthropic/`, `gemini/`) and LiteLLM will require the corresponding API keys in the environment. LiteLLM also supports custom settings (e.g., safety configs as shown for Gemini or system prompts).

We then run the LiteLLM proxy server as a Docker container (see Deployment section). The FastAPI router will communicate with it. For example, the helper function `call_model_via_litellm` might be implemented with HTTP calls like:

```python
import os, requests, json
LITELLM_URL = os.getenv("LITELLM_URL", "http://litellm:4000/v1")  # litellm service in docker
LITELLM_KEY = os.getenv("LITELLM_MASTER_KEY", "your_master_key")

def call_model_via_litellm(model_name, messages, stream=False):
    headers = {"Authorization": f"Bearer {LITELLM_KEY}"}
    payload = {"model": model_name, "messages": messages, "stream": stream}
    resp = requests.post(f"{LITELLM_URL}/chat/completions", headers=headers, json=payload, stream=stream)
    if stream:
        # Return a generator over response chunks
        def chunk_generator():
            for line in resp.iter_lines():
                if line:  # filter keep-alive newlines
                    decoded = line.decode("utf-8")
                    if decoded.strip().startswith("data: "):
                        data = decoded[len("data: "):]
                        if data == "[DONE]":
                            break
                        yield json.loads(data)["choices"][0]["delta"].get("content", "")
        return chunk_generator()
    else:
        result = resp.json()
        return result["choices"][0]["message"]["content"]
```

The above uses the OpenAI-compatible REST API exposed by LiteLLM (which by default listens on port 4000). If `stream=True`, LiteLLM will itself stream SSE data lines; we capture those and yield the tokens. If not streaming, we just parse the JSON response.

**Ollama Setup:** We will ensure an Ollama daemon is running (in Docker or on host) serving the local model specified (e.g., `ollama pull CodeLlama-13B-Python` beforehand). Alternatively, LiteLLM has a Docker image that includes Ollama support (the `litellm/ollama` image can run a local server with certain models). In our Docker Compose, we might run a separate Ollama container or use the LiteLLM container’s ability to launch it.

Using local models for straightforward or smaller tasks can drastically cut costs – as noted by Switchpoint, this approach can yield *“up to 95% cost savings”* when GPT-4 calls are avoided. The trade-off is that local models might occasionally produce lower quality output, which is why we have the fallback logic and continuous learning to judge when to use them.

## Reinforcement Learning Loop (DSPy Self-Optimization)

Initially, our routing rules are based on heuristics and our best guesses of model suitability. However, to truly optimize cost and performance, the system should **learn from experience**. We integrate a reinforcement learning (RL) loop using DSPy’s optimization capabilities to fine-tune the routing policy over time.

**Feedback and Logging:** Each interaction with the system can provide a learning signal. We log:

* Which model(s) were used for a request.
* The outcome: Did the user accept the answer? Did they need a second attempt or re-route? Possibly we can infer outcome by user behavior (if they immediately ask a follow-up like “That didn’t work, try again”, we infer failure).
* Explicit feedback if available (for example, if this system is used internally, developers might mark answers as helpful or not).
* Cost metrics (tokens consumed, latency). We can compute a *reward* for the decision: for instance, **reward = (success\_score \* 1.0) – (cost\_factor \* 0.01) – (latency\_penalty)**. A successful completion that uses a cheap model yields a high reward, whereas a failure or using an expensive model yields a lower reward.

These logs form a dataset for training. The RL problem can be framed as a **contextual bandit** or sequential decision problem where the context is the query (or features of it), and the action is the model choice (or sequence of choices), and the reward is as defined above.

**DSPy RL Training:** DSPy provides an algorithm called **GRPO** (Generalized Reweighted Policy Optimization) designed for online RL with language model programs. We can create a DSPy **policy module** for routing. For example, imagine we have a small model or a parameter in our router that decides between “local” vs “cloud” model for a given intent. We can represent that decision as a soft parameter (like a probability or a weight in a sigmoid). DSPy would allow us to **optimize that parameter** to maximize the reward over our logged experiences.

In practice, the RL loop might work as follows:

* We periodically run a training job (`trainer/train.py`) that reads the recent usage log.
* It constructs training examples: each example includes the input features (could be the text of the prompt, or some encoded representation such as length, presence of keywords, etc.), the action taken (model chosen), and the outcome reward.
* We use DSPy to define the router policy with *tunable parameters*. For instance, if our router uses a classifier or a logistic regression to pick a model, the weights of that classifier are what we want to tune.
* The trainer then runs `dspy.optimize` with an RL algorithm (like GRPO or even simpler bandit optimization if appropriate) on these examples to update the policy. DSPy’s GRPO is specifically built to handle optimizing *multi-module programs* like ours, treating the entire decision process as a computation graph and adjusting it.

**RL Training Code Snippet (conceptual):**

```python
# trainer/train.py
import dspy
from router.router import CodeRouterModule  # hypothetical DSPy Module for router policy

# Load logged data
training_data = load_usage_log("usage_log.jsonl")
# Each entry in training_data might have: {"features": ..., "chosen_model": ..., "reward": ...}

# Initialize the router policy module (with current weights)
router_module = CodeRouterModule()

# Define an optimizer – using DSPy's GRPO for RL
optimizer = dspy.GRPO(router_module, epochs=5, lr=0.01)

# Convert training data into DSPy format (inputs and reward labels)
train_inputs = [d["features"] for d in training_data]
train_rewards = [d["reward"] for d in training_data]

# Train the policy module to maximize expected reward
optimizer.train(train_inputs, train_rewards)

# After training, save the updated module parameters (or the whole module)
router_module.save("router_policy.dspy")
```

In this pseudo-code:

* `CodeRouterModule` would be a DSPy module that encapsulates the decision of which model to use. It might output a distribution over model options or directly a selection. Its internal parameters (e.g., weights for a neural net or just a few scalars for thresholds) are what we train.
* `dspy.GRPO` is initialized with our module; we run a few epochs of training using the collected data.
* We then save the tuned policy. In deployment, the router would load these learned parameters on startup (so that the next time it makes decisions, it uses the improved policy).

The learning is **extensible**: as new models become available or as user behavior changes, the RL will adjust. For example, if we introduce a new model (say a more efficient local model), initially the system might not use it heavily, but if it proves to solve tasks with equal success at lower cost (thus higher reward), the RL optimizer will increase the policy’s preference for that model in relevant situations.

Moreover, DSPy’s approach treats the LLM routing as a **modular program** rather than a black-box prompt. This means we can improve parts of it (like the intent classifier or the fallback criteria) using data. Over time, the router may learn nuanced behaviors, such as recognizing specific types of errors that the local model often fails at and directly using GPT-4 for those (because historically that yielded higher reward).

Finally, we can incorporate **user feedback** explicitly: if users can rate the helpfulness of an answer, that can directly influence the reward. A thumbs-up could be +1 reward, a thumbs-down -1, etc., guiding the RL to favor models that users prefer for certain tasks.

## Project Structure and Deployment

We will organize the project into a clear folder structure, include all necessary requirements, and provide Docker support for easy deployment. Below is the structure:

```bash
switchpoint-code-router/
├── gateway/
│   ├── main.py            # FastAPI app defining the API endpoints
│   └── Dockerfile         # Container for gateway (installs requirements and runs app)
├── router/
│   ├── router.py          # Routing logic (intent classification and model selection)
│   ├── intents.py         # (Optional) intent detection helper (regex or small model)
│   └── policy.dspy        # (Optional) saved DSPy policy parameters for the router
├── trainer/
│   ├── train.py           # RL training script to optimize routing policy
│   └── reward_scheme.py   # Definition of reward computation from outcomes
├── models/
│   └── prompts/           # (Optional) prompt templates or few-shot examples for classification
├── litellm_config.yaml    # Configuration for LiteLLM proxy (model endpoints)
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Compose file to run gateway, liteLLM (and optionally trainer)
└── README.md              # Documentation on setup and usage
```

**Key configuration files:**

* `requirements.txt` will include:

  * FastAPI and Uvicorn (`fastapi`, `uvicorn[standard]`) for the API.
  * DSPy (`dspy`) for the router logic and training.
  * LiteLLM Python SDK (`litellm`) if we use it in code, or just use HTTP calls.
  * Requests (`requests`) for calling LiteLLM or other HTTP APIs.
  * Any model SDKs if directly used (e.g., `openai`, `anthropic`, but LiteLLM can abstract those).
  * Other utilities like `pydantic` (FastAPI uses it), `numpy/pandas` (if used in trainer), etc.

Example **requirements.txt**:

```
fastapi==0.100.0
uvicorn[standard]==0.22.0
dspy==0.3.0        # (example version)
litellm==0.2.5     # LiteLLM SDK for Python
requests==2.31.0
```

**Docker Compose Setup:**

We use Docker Compose to run at least two services: the API gateway and the LiteLLM proxy. Optionally, we could containerize the trainer or run it as a one-off job. Here’s a simplified `docker-compose.yml`:

```yaml
version: "3.9"
services:
  gateway:
    build: ./gateway
    ports:
      - "8080:8080"                # expose FastAPI on port 8080
    environment:
      - LITELLM_URL=http://litellm:4000/v1
      - LITELLM_MASTER_KEY=dummy_key        # match LiteLLM's configured master key
    depends_on:
      - litellm

  litellm:
    image: ghcr.io/berriai/litellm:main-latest   # official LiteLLM Proxy image
    command: ["--config=/litellm_config.yaml", "--detailed_debug"]
    ports:
      - "4000:4000"              # expose LiteLLM (could be only internal)
    environment:
      - LITELLM_MASTER_KEY=dummy_key
      - OPENAI_API_KEY=${OPENAI_API_KEY}        # pulled from .env or environment
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      # (Include any other provider keys needed)
    volumes:
      - ./litellm_config.yaml:/litellm_config.yaml:ro   # mount our model config
```

In this configuration:

* The **gateway** service builds from the `gateway/` directory. The `gateway/Dockerfile` would install Python, copy the code and requirements, then run `uvicorn gateway.main:app --host 0.0.0.0 --port 8080`.
* The **litellm** service uses the pre-built image (for convenience, as it already contains the proxy server). We pass our config file and required API keys. `LITELLM_MASTER_KEY` is set to a known value (“dummy\_key” here) which our gateway will use for authorization when calling the LiteLLM API.
* We expose port 8080 for clients to call our FastAPI, and port 4000 for LiteLLM. (If you want to keep LiteLLM internal only, you could omit exposing 4000, since gateway and litellm communicate on the internal Docker network. Here we expose it for potential direct access or debugging.)

After filling in the real API keys (in an `.env` file or environment variables), deployment is as simple as:

```bash
docker-compose up --build
```

This will start the gateway and the LiteLLM proxy. The trainer is not started by default here; you can run it manually when you want to update the policy (for example: `docker-compose run --rm gateway python trainer/train.py` to execute training inside the gateway container, assuming it has the logs accessible).

**Dockerfile for gateway:** (an example to complete the picture)

```dockerfile
# gateway/Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY ../requirements.txt ./ 
RUN pip install -r requirements.txt
COPY . . 
# Set environment (if needed, e.g., PYTHONUNBUFFERED for logging)
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

We mount `litellm_config.yaml` read-only into the LiteLLM container to specify our models. That config plus the API keys environment will let LiteLLM know how to route model requests:

* For `model_name: local-code-model`, it will connect to Ollama. (We might need to ensure Ollama is running. If the LiteLLM image includes an Ollama backend, it might manage it; otherwise, we might run an Ollama daemon separately. One simple approach: install Ollama on the host and run `ollama serve`, then LiteLLM will connect to it at `localhost:11434` by default. Alternatively, incorporate an Ollama container and point LiteLLM to it via config.)

**Scaling and Performance:** Each component can be scaled. The FastAPI can be replicated behind a load balancer if needed. LiteLLM can be scaled or pointed to a cluster of model workers if many requests or heavy models. Because we stream responses, ensure to configure timeouts appropriately (some model calls, especially large ones, might take many seconds).

In summary, this design provides a robust, extensible clone of Switchpoint.dev for code-related queries. It intelligently routes requests to the cheapest or fastest model that can handle them (leveraging local models for simple tasks and powerful cloud models for hard tasks), and it continuously learns from experience to improve its routing decisions using DSPy’s reinforcement learning framework. By containerizing the solution and using standard interfaces, it’s easy to deploy and integrate with existing development workflows – developers can simply point their tools at our FastAPI endpoint and benefit from optimized LLM routing without any manual model selection.

**Sources:**

* Switchpoint AI (Michaelson, 2025) – Intelligent LLM routing saving cost via model selection
* LiteLLM Documentation – OpenAI-compatible proxy for 100+ local/remote LLMs
* DSPy Project – Framework for modular LLM programs and self-optimization with RL (GRPO algorithm for policy learning)


**[➡️ REPLACE: Write a clear, concise description of your MCP server's purpose. What problems does it solve? What capabilities does it provide to AI tools?]**

## Overview

**[➡️ REPLACE: Provide a more detailed explanation of your MCP server's architecture, key components, and how it integrates with AI tools. What makes it unique or valuable?]**

## Features

**[➡️ REPLACE: List the key features of your MCP server. Some examples:]
- What unique tools does it provide?
- What data sources can it access?
- What special capabilities does it have?
- What performance characteristics are notable?
- What integrations does it support?**

## Installation

### From PyPI (if published)

```bash
# Install using UV (recommended)
uv pip install dspy_mcp

# Or using pip
pip install dspy_mcp
```

### From Source

```bash
# Clone the repository
git clone <your-repository-url>
cd dspy_mcp

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
uv pip install -e .
```

**[➡️ REPLACE: Update installation instructions. If you plan to publish to PyPI, keep the PyPI section. Otherwise, remove it and focus on source installation.]**

## Available Tools

### tool_name

**[➡️ REPLACE: For each tool in your MCP server, document:]
- What the tool does
- Its parameters and their types
- What it returns
- Example usage and expected output
- Any limitations or important notes**

Example:
```bash
# Using stdio transport (default)
dspy_mcp-client "your command here"

# Using SSE transport
dspy_mcp-server --transport sse
curl http://localhost:3001/sse
```

## Usage

This MCP server provides two entry points:

1. `dspy_mcp-server`: The MCP server that handles tool requests
   ```bash
   # Run with stdio transport (default)
   dspy_mcp-server

   # Run with SSE transport
   dspy_mcp-server --transport sse
   ```

## Logging

The server logs all activity to both stderr and a rotating log file. Log files are stored in OS-specific locations:

- **macOS**: `~/Library/Logs/mcp-servers/dspy_mcp.log`
- **Linux**: 
  - Root user: `/var/log/mcp-servers/dspy_mcp.log`
  - Non-root: `~/.local/state/mcp-servers/logs/dspy_mcp.log`
- **Windows**: `%USERPROFILE%\AppData\Local\mcp-servers\logs\dspy_mcp.log`

Log files are automatically rotated when they reach 10MB, with up to 5 backup files kept.

You can configure the log level using the `LOG_LEVEL` environment variable:
```bash
# Set log level to DEBUG for more detailed logging
LOG_LEVEL=DEBUG dspy_mcp-server
```

Valid log levels are: DEBUG, INFO (default), WARNING, ERROR, CRITICAL

2. `dspy_mcp-client`: A convenience client for testing
   ```bash
   dspy_mcp-client "your command here"
   ```

**[➡️ REPLACE: Add any additional usage examples, common patterns, or best practices specific to your tools]**

## Requirements

- Python 3.11 or later (< 3.13)
- Operating Systems: Linux, macOS, Windows

**[➡️ REPLACE: Add any additional requirements specific to your MCP server:]
- Special system dependencies
- External services or APIs needed
- Network access requirements
- Hardware requirements (if any)**

## Configuration

**[➡️ REPLACE: Document any configuration options your MCP server supports:]
- Environment variables
- Configuration files
- Command-line options
- API keys or credentials needed

Remove this section if your server requires no configuration.**

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development instructions.

**[➡️ REPLACE: Add any project-specific development notes, guidelines, or requirements]**

## Troubleshooting

Common issues and their solutions:

**[➡️ REPLACE: Add troubleshooting guidance specific to your MCP server. Remove this section if not needed.]**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

**[➡️ REPLACE: Add your name and contact information]**

---

[Replace this example Echo server README with documentation specific to your MCP server. Use this structure as a template, but customize all sections to describe your server's actual functionality, tools, and configuration options.]
