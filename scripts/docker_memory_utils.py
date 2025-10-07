#!/usr/bin/env python3

"""
Docker memory optimization utilities for Python scripts.
This module provides functions to get memory-optimized Docker run flags.
"""

import subprocess
import os
import sys


def get_docker_memory_flags():
    """
    Get memory optimization flags for Docker run commands.
    
    Returns:
        str: Memory optimization flags (e.g., "--memory=200G --memory-swap=200G --oom-kill-disable")
    """
    # Allow override via environment variable
    if 'KPFPIPE_MEMORY_LIMIT_GB' in os.environ:
        memory_limit = os.environ['KPFPIPE_MEMORY_LIMIT_GB']
        return f"--memory={memory_limit}G --memory-swap={memory_limit}G --oom-kill-disable"
    
    # Get system memory info
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        # Parse total and available memory
        total_kb = None
        available_kb = None
        
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                total_kb = int(line.split()[1])
            elif line.startswith('MemAvailable:'):
                available_kb = int(line.split()[1])
        
        if total_kb is None or available_kb is None:
            # Fallback to default
            return "--memory=4G --memory-swap=4G --oom-kill-disable"
        
        # Convert to GB
        total_gb = total_kb // (1024 * 1024)
        available_gb = available_kb // (1024 * 1024)
        
        # Calculate suggested memory allocation
        if total_gb <= 16:
            # Laptop/small system: use 40% of available RAM, max 6GB (for 2 containers)
            suggested_ram = min(available_gb * 40 // 100, 6)
        elif total_gb <= 64:
            # Medium system: use 45% of available RAM, max 24GB (for 2 containers)
            suggested_ram = min(available_gb * 45 // 100, 24)
        elif total_gb <= 256:
            # Large system: use 50% of available RAM, max 64GB (for 4 containers)
            suggested_ram = min(available_gb * 50 // 100, 64)
        else:
            # Very large system: use 60% of available RAM, max 200GB (for 8+ containers)
            suggested_ram = min(available_gb * 60 // 100, 200)
        
        # Minimum 4GB for basic operations
        suggested_ram = max(suggested_ram, 4)
        
        return f"--memory={suggested_ram}G --memory-swap={suggested_ram}G --oom-kill-disable"
        
    except Exception as e:
        # Fallback to default on any error
        print(f"Warning: Could not determine optimal memory allocation: {e}", file=sys.stderr)
        return "--memory=4G --memory-swap=4G --oom-kill-disable"


def build_docker_run_cmd(base_cmd, memory_flags=None):
    """
    Build a Docker run command with memory optimization.
    
    Args:
        base_cmd (str): Base Docker run command
        memory_flags (str, optional): Memory flags. If None, will be calculated automatically.
    
    Returns:
        str: Complete Docker run command with memory optimization
    """
    if memory_flags is None:
        memory_flags = get_docker_memory_flags()
    
    # Insert memory flags after "docker run" but before other arguments
    if base_cmd.startswith('docker run'):
        # Find the first argument after "docker run"
        parts = base_cmd.split(' ', 2)
        if len(parts) >= 3:
            return f"{parts[0]} {parts[1]} {memory_flags} {parts[2]}"
        else:
            return f"{base_cmd} {memory_flags}"
    else:
        return f"{base_cmd} {memory_flags}"


if __name__ == '__main__':
    # When run as script, just output the memory flags
    print(get_docker_memory_flags())
