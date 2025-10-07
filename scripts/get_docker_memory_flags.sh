#!/bin/bash

# Helper script to get memory optimization flags for Docker commands
# This can be called from Perl scripts to get the appropriate memory flags

# Source the shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/docker_launch_utils.sh"

# Get the memory limit
memory_limit_gb=$(calculate_memory_limit)

# Output the memory flags in a format that can be easily used in Perl
echo "--memory=${memory_limit_gb}G --memory-swap=${memory_limit_gb}G --oom-kill-disable"
