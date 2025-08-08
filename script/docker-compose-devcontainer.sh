#!/bin/bash

# Helper script to run docker compose from within a devcontainer
# It automatically detects the host path and sets the environment variable

set -e

# Function to detect host path
detect_host_path() {
    # Try multiple methods to detect the host path
    local host_path=""
    
    # Method 1: Parse mountinfo for /workspace mount
    host_path=$(cat /proc/self/mountinfo | grep " /workspace " | awk '{print $4}' | head -1)
    
    if [ -z "$host_path" ]; then
        # Method 2: Check for bind mount specifically
        host_path=$(cat /proc/self/mountinfo | grep "/workspace " | grep "bind" | awk '{print $4}' | head -1)
    fi
    
    echo "$host_path"
}

# Check if we're in a devcontainer environment
if [ -f /.dockerenv ] || [ -n "$REMOTE_CONTAINERS" ] || [ -n "$CODESPACES" ]; then
    echo "Detected devcontainer environment"
    
    # Try to detect host path
    HOST_PATH=$(detect_host_path)
    
    if [ -z "$HOST_PATH" ]; then
        echo "Warning: Could not detect host path automatically"
        echo "Please set HOST_PROJECT_PATH manually:"
        echo "  export HOST_PROJECT_PATH=/your/host/project/path"
        echo "  docker compose $*"
        exit 1
    fi
    
    echo "Host project path: $HOST_PATH"
    export HOST_PROJECT_PATH="$HOST_PATH"
    
    # Run docker compose with the environment variable set
    # The docker-compose.override.yml file will use this variable
    exec docker compose "$@"
else
    echo "Not in devcontainer, running docker compose with default paths"
    exec docker compose "$@"
fi