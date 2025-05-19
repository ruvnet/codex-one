import os
import sys

# Ensure the parent directory is on the path so tests can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
