#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define the name of the virtual environment directory
VENV_DIR=".venv"

# Define the path to the requirements file
REQUIREMENTS_FILE="requirements.txt"

# Function to display messages
function echo_msg() {
    echo -e "\n\033[1;32m$1\033[0m\n"
}

# Check if Python 3.11 or higher is installed
if ! python3 --version | grep -qE "Python 3\.(1[1-9]|[2-9][0-9])"; then
    echo_msg "Error: Python 3.11 or higher is required."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo_msg "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo_msg "Virtual environment already exists in $VENV_DIR."
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip to the latest version
echo_msg "Upgrading pip..."
pip install --upgrade pip

# Install required packages from requirements.txt
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo_msg "Installing packages from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo_msg "Error: $REQUIREMENTS_FILE not found."
    deactivate
    exit 1
fi

# Display success message
echo_msg "Setup complete. Virtual environment is ready."

# Deactivate the virtual environment
deactivate