#!/bin/bash

# Set up signal handling for graceful shutdown
trap 'echo "Shutting down Redis..."; kill $REDIS_PID 2>/dev/null || true; exit 0' SIGTERM SIGINT

# Configure kernel settings to eliminate Redis warnings
echo "Configuring kernel settings for Redis..."
# Try to set overcommit_memory (requires root or proper permissions)
if [ -w /proc/sys/vm/overcommit_memory ]; then
    echo 1 > /proc/sys/vm/overcommit_memory
    echo "✅ Set overcommit_memory = 1"
fi

# Try to disable transparent huge pages
if [ -w /sys/kernel/mm/transparent_hugepage/enabled ]; then
    echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
    echo "✅ Set THP = madvise"
fi

# Start Redis in the background
echo "Starting Redis server..."

# Try to start Redis on default port, redirecting all output to suppress warnings
redis-server /etc/redis/redis.conf > /dev/null 2>&1 &
REDIS_PID=$!

# Wait a moment for Redis to start
sleep 2

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Redis failed to start on default port, trying alternative port..."
    # Kill the failed process
    kill $REDIS_PID 2>/dev/null || true
    
    # Try alternative port 6380
    redis-server --port 6380 /etc/redis/redis.conf > /dev/null 2>&1 &
    REDIS_PID=$!
    sleep 2
    
    if ! redis-cli -p 6380 ping > /dev/null 2>&1; then
        echo "Warning: Redis failed to start on alternative port, continuing without caching..."
    else
        echo "Redis server started successfully on port 6380 (PID: $REDIS_PID)"
        # Update Redis client to use alternative port
        export REDIS_PORT=6380
    fi
else
    echo "Redis server started successfully on default port (PID: $REDIS_PID)"
fi

# Handle different command types
if [ "$1" = "make" ]; then
    # If make is requested, run it directly
    exec "$@"
elif [ "$1" = "bash" ]; then
    # If bash is requested, just run it
    exec "$@"
elif [ "$1" = "sh" ]; then
    # If sh is requested, just run it
    exec "$@"
elif [ "$1" = "redis-cli" ]; then
    # If redis-cli is requested, run it
    exec "$@"
else
    # Default: run python with the given arguments
    exec python "$@"
fi
