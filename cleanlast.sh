#!/bin/bash

echo "Stopping TorchServe..."
torchserve --stop

echo "Checking for remaining Java processes..."
ps aux | grep java
ps aux | grep torch

echo "Removing PID file..."
rm -f /tmp/.model_server.pid

echo "Checking for processes using port 8080..."
PORT_PROCS=$(lsof -i :8080 -t)
if [ ! -z "$PORT_PROCS" ]; then
    echo "Killing processes using port 8080: $PORT_PROCS"
    kill -9 $PORT_PROCS
fi

echo "Checking port 8081 and 8082 used by TorchServe management & metrics..."
PORT_PROCS_8081=$(lsof -i :8081 -t)
if [ ! -z "$PORT_PROCS_8081" ]; then
    echo "Killing processes using port 8081: $PORT_PROCS_8081"
    kill -9 $PORT_PROCS_8081
fi

PORT_PROCS_8082=$(lsof -i :8082 -t)
if [ ! -z "$PORT_PROCS_8082" ]; then
    echo "Killing processes using port 8082: $PORT_PROCS_8082"
    kill -9 $PORT_PROCS_8082
fi

echo "Killing any Java processes with TorchServe..."
pkill -f "java.*torch"

echo "Setting environment variable for CUDA libraries..."
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo "TorchServe cleanup complete."