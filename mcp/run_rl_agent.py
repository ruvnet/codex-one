#!/usr/bin/env python
"""
Script to run and train the DSPy RL agent in the MCP server
"""

import argparse
import sys
import json
import requests
import time
import os
import atexit
import subprocess
from typing import List, Tuple, Dict, Any

# Ensure we can import from the mcp module
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dspy_mcp.pipeline.rl_pipeline import RLEchoAgent, train_agent


def start_server(port: int = 3001) -> subprocess.Popen:
    """Start the MCP server in SSE mode"""
    print(f"Starting MCP server on port {port}...")
    
    # Start the server process
    server_process = subprocess.Popen(
        ["dspy_mcp-server", "--transport", "sse", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Register cleanup function to kill server on exit
    def cleanup():
        if server_process.poll() is None:
            print("Shutting down server...")
            server_process.terminate()
            server_process.wait(timeout=5)
    
    atexit.register(cleanup)
    
    # Wait for server to start
    time.sleep(2)
    
    # Verify server is running
    try:
        response = requests.get(f"http://localhost:{port}/")
        if response.status_code != 200:
            print(f"Server response code: {response.status_code}")
            raise Exception("Server not responding correctly")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        cleanup()
        sys.exit(1)
    
    print("Server started successfully")
    return server_process


def call_echo_tool(text: str, transform: str = None, port: int = 3001) -> Dict[str, Any]:
    """Call the echo tool on the MCP server"""
    url = f"http://localhost:{port}/tools/echo"
    payload = {"text": text}
    if transform:
        payload["transform"] = transform
    
    response = requests.post(url, json=payload)
    return response.json()


def collect_training_data(num_samples: int = 10, port: int = 3001) -> List[Tuple[str, str, float]]:
    """
    Collect training data by calling the echo tool with different transforms
    and assigning rewards based on simple criteria
    """
    training_data = []
    sample_texts = [
        "Hello, world!",
        "Reinforcement learning is fun",
        "DSPy makes RL easier",
        "MCP server in action",
        "Training an echo agent",
    ]
    
    for _ in range(num_samples):
        # Choose a random text
        text = sample_texts[_ % len(sample_texts)]
        
        # Try both upper and lower transforms
        for transform in ["upper", "lower"]:
            result = call_echo_tool(text, transform, port)
            
            # Assign reward based on some criteria
            # For demonstration, we'll reward uppercase for exclamations and lowercase otherwise
            response_text = result.get("text", "")
            reward = 1.0 if (transform == "upper" and "!" in text) or (transform == "lower" and "!" not in text) else -0.5
            
            training_data.append((text, transform, reward))
            print(f"Sample: '{text}' → '{response_text}' ({transform}) → Reward: {reward}")
    
    return training_data


def train_rl_agent(training_data: List[Tuple[str, str, float]]):
    """Train the RL agent using collected data"""
    print(f"\nTraining RL agent with {len(training_data)} samples...")
    
    # Create and train agent
    agent = RLEchoAgent()
    initial_prob = agent.prob_upper()
    print(f"Initial probability of choosing 'upper': {initial_prob:.4f}")
    
    # Train the agent
    train_agent(agent, training_data)
    
    # Report results
    final_prob = agent.prob_upper()
    print(f"Final probability of choosing 'upper': {final_prob:.4f}")
    print(f"Delta: {final_prob - initial_prob:.4f}")
    
    return agent


def test_trained_agent(agent: RLEchoAgent, num_tests: int = 5):
    """Test the trained agent to see what actions it takes"""
    print("\nTesting trained agent...")
    test_texts = [
        "This is a test!",
        "No exclamation here",
        "Another exclamation mark!",
        "Plain text",
        "Exciting test!"
    ]
    
    for i in range(min(num_tests, len(test_texts))):
        text = test_texts[i]
        prediction = agent(text=text)
        print(f"Input: '{text}'")
        print(f"Action chosen: {prediction.action}")
        print(f"Response: '{prediction.response}'")
        print()


def main():
    """Main function to run the RL pipeline"""
    parser = argparse.ArgumentParser(description="Run and train the DSPy RL agent in the MCP server")
    parser.add_argument("--port", type=int, default=3001, help="Port for the MCP server")
    parser.add_argument("--samples", type=int, default=10, help="Number of training samples to collect")
    parser.add_argument("--no-server", action="store_true", help="Don't start a new server (use existing one)")
    args = parser.parse_args()
    
    # Start server if needed
    server_process = None
    if not args.no_server:
        server_process = start_server(args.port)
    
    try:
        # Collect training data
        print(f"\nCollecting {args.samples} training samples...")
        training_data = collect_training_data(args.samples, args.port)
        
        # Train the agent
        agent = train_rl_agent(training_data)
        
        # Test the trained agent
        test_trained_agent(agent)
        
        print("\nRL agent training complete!")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup happens through atexit handler
        pass


if __name__ == "__main__":
    main()