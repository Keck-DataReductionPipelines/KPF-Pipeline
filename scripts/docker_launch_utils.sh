#!/bin/bash

# Common Docker launch utilities for KPF-Pipeline
# This script provides memory-optimized Docker container launching functions

# Function to calculate optimal memory allocation
calculate_memory_limit() {
    # Allow override via environment variable
    if [ -n "$KPFPIPE_MEMORY_LIMIT_GB" ]; then
        echo "$KPFPIPE_MEMORY_LIMIT_GB"
        return
    fi
    
    local total_ram_gb=$(free -g | awk '/^Mem:/{print $2}')
    local available_ram_gb=$(free -g | awk '/^Mem:/{print $7}')
    
    # Adaptive memory allocation based on system size
    # Conservative for production (8+ containers) vs development (1-2 containers)
    local suggested_ram
    
    if [ $total_ram_gb -le 16 ]; then
        # Laptop/small system: use 40% of available RAM, max 6GB (for 2 containers)
        suggested_ram=$((available_ram_gb * 40 / 100))
        if [ $suggested_ram -gt 6 ]; then
            suggested_ram=6
        fi
    elif [ $total_ram_gb -le 64 ]; then
        # Medium system: use 45% of available RAM, max 24GB (for 2 containers)
        suggested_ram=$((available_ram_gb * 45 / 100))
        if [ $suggested_ram -gt 24 ]; then
            suggested_ram=24
        fi
    elif [ $total_ram_gb -le 256 ]; then
        # Large system: use 50% of available RAM, max 64GB (for 4 containers)
        suggested_ram=$((available_ram_gb * 50 / 100))
        if [ $suggested_ram -gt 64 ]; then
            suggested_ram=64
        fi
    else
        # Very large system (like your 2TB): use 60% of available RAM, max 200GB (for 8+ containers)
        suggested_ram=$((available_ram_gb * 60 / 100))
        if [ $suggested_ram -gt 200 ]; then
            suggested_ram=200
        fi
    fi
    
    # Minimum 4GB for basic operations
    if [ $suggested_ram -lt 4 ]; then
        suggested_ram=4
    fi
    
    echo "$suggested_ram"
}

# Function to build optimized Docker run command
build_docker_run_cmd() {
    local base_cmd="$1"
    local memory_limit_gb=$(calculate_memory_limit)
    
    # Add memory optimization flags
    local optimized_cmd="${base_cmd} --memory=${memory_limit_gb}G --memory-swap=${memory_limit_gb}G"
    
    echo "$optimized_cmd"
}

# Function to launch Docker container with memory optimization
launch_docker_container() {
    local container_name="$1"
    local image="$2"
    local base_args="$3"
    local command="$4"
    
    local memory_limit_gb=$(calculate_memory_limit)
    
    echo "Launching container '$container_name' with ${memory_limit_gb}GB RAM allocation (swap disabled)"
    
    # Build the full command
    local full_cmd="docker run -d --name $container_name $base_args --memory=${memory_limit_gb}G --memory-swap=${memory_limit_gb}G $image $command"
    
    echo "Executing: $full_cmd"
    eval "$full_cmd"
}

# Export functions for use in other scripts
export -f calculate_memory_limit
export -f build_docker_run_cmd
export -f launch_docker_container
