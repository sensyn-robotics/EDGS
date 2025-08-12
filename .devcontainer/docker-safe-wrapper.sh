#!/bin/bash

# Docker Safe Wrapper for DevContainers
# This wrapper allows only safe Docker commands to prevent accidental system damage
# It's designed to be generic and distributable across projects

# Define safe commands patterns
SAFE_PATTERNS=(
    # Allow viewing container/image information
    "^docker ps"
    "^docker images"
    "^docker version"
    "^docker info"
    "^docker logs"
    "^docker inspect"
    
    # Allow docker compose read operations
    "^docker compose ps"
    "^docker compose logs"
    "^docker compose version"
    "^docker compose ls"
    "^docker compose config"
    
    # Allow docker-compose (hyphenated) read operations  
    "^docker-compose ps"
    "^docker-compose logs"
    "^docker-compose version"
    "^docker-compose ls"
    "^docker-compose config"
    
    # Allow executing into existing containers (any name)
    "^docker exec"
    "^docker compose exec"
    "^docker-compose exec"
    
    # Allow starting/stopping existing services
    "^docker compose up"
    "^docker compose down"
    "^docker compose stop"
    "^docker compose start"
    "^docker compose restart"
    "^docker-compose up"
    "^docker-compose down"
    "^docker-compose stop"
    "^docker-compose start"
    "^docker-compose restart"
)

# Commands that are explicitly blocked
DANGEROUS_PATTERNS=(
    # Prevent system-wide operations
    "system prune"
    "container prune"
    "image prune"
    "volume prune"
    "network prune"
    
    # Prevent removing volumes (data loss)
    "volume rm"
    "volume remove"
    
    # Prevent building (can consume resources)
    "build"
    "buildx"
    
    # Prevent pushing to registries
    "push"
    "tag"
)

# Construct the command string
COMMAND="docker $*"

# Check for dangerous patterns first
for pattern in "${DANGEROUS_PATTERNS[@]}"; do
    if [[ "$COMMAND" == *"$pattern"* ]]; then
        echo "❌ Error: This Docker command contains dangerous operations that are not allowed in devcontainer."
        echo "   Command blocked: $pattern"
        echo ""
        echo "💡 Tip: If you need to perform system maintenance, run this command from the host machine."
        exit 1
    fi
done

# Check if command matches safe patterns
IS_SAFE=false
for pattern in "${SAFE_PATTERNS[@]}"; do
    if [[ "$COMMAND" =~ $pattern ]]; then
        IS_SAFE=true
        break
    fi
done

if [ "$IS_SAFE" = true ]; then
    # Execute the actual docker command
    /usr/bin/docker "$@"
else
    echo "⚠️  Warning: This Docker command is restricted in the devcontainer for safety."
    echo "   Command: $COMMAND"
    echo ""
    echo "✅ Allowed operations:"
    echo "   • View: ps, images, logs, inspect, version, info"
    echo "   • Compose: up, down, stop, start, restart, exec, logs"
    echo "   • Execute: docker exec into any container"
    echo ""
    echo "💡 If you need to run this specific command, use it from the host machine."
    exit 1
fi