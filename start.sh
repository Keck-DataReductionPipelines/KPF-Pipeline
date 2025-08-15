#!/bin/bash

# Configure kernel settings to eliminate Redis warnings
echo "Configuring kernel settings for Redis..."
# Try to set overcommit_memory (requires root or proper permissions)
if [ -w /proc/sys/vm/overcommit_memory ]; then
    echo 1 > /proc/sys/vm/overcommit_memory
    echo "✅ Set overcommit_memory = 1"
else
    echo "⚠️  Cannot set overcommit_memory (requires root permissions)"
fi

# Try to disable transparent huge pages
if [ -w /sys/kernel/mm/transparent_hugepage/enabled ]; then
    echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
    echo "✅ Set THP = madvise"
else
    echo "⚠️  Cannot set THP (requires root permissions)"
fi

# Start Redis in the background
echo "Starting Redis server..."
redis-server /etc/redis/redis.conf &
REDIS_PID=$!

# Wait a moment for Redis to start
sleep 2

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Warning: Redis failed to start, continuing without caching..."
else
    echo "Redis server started successfully (PID: $REDIS_PID)"
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
