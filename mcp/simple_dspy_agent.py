#!/usr/bin/env python3
"""
Simple script to demonstrate using the DSPy agent in the MCP server.
"""

import os
import sys
import argparse
from dspy_mcp.pipeline.agent_pipeline import run_agent, echo_agent
from dspy_mcp.tools.echo import echo

def main():
    """Main function to run the DSPy agent directly."""
    parser = argparse.ArgumentParser(description="Run the DSPy agent directly")
    parser.add_argument("text", help="Text to process")
    parser.add_argument("--transform", choices=["upper", "lower"], 
                      help="Optional transformation to apply")
    args = parser.parse_args()

    print(f"\n--- Direct Echo Tool ---")
    result = echo(args.text, args.transform)
    print(f"Result: {result.text}")
    
    print(f"\n--- DSPy Echo Agent ---")
    agent_result = run_agent(args.text, args.transform)
    print(f"Result: {agent_result}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())