#!/bin/bash
# Environment setup script for LidarProjectionLane project
# This script configures paths, dependencies, and development environment

# Exit on any error
set -e

echo "Setting up LidarProjectionLane development environment..."

# Directory where script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configure environment variables
export LANE_FUSION_ROOT=$PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/scripts

# Setup CUDA paths if not already set
if [ -z "$CUDA_HOME" ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi

# Check for YOLO model
if [ ! -f "$PROJECT_ROOT/lane_detection/models/yolov8n-seg-lane.engine" ]; then
    echo "YOLOv8 model not found. You'll need to convert it using the provided scripts."
    echo "Run: python3 $PROJECT_ROOT/scripts/convert_yolo_model.py when you have the model ready."
fi

# Check development dependencies
echo "Checking dependencies..."
MISSING_DEPS=0

# Check for essential development tools
command -v cmake >/dev/null 2>&1 || { echo "cmake not found"; MISSING_DEPS=1; }
command -v nvcc >/dev/null 2>&1 || { echo "CUDA nvcc not found"; MISSING_DEPS=1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; MISSING_DEPS=1; }

# Check for ROS
if [ -z "$ROS_DISTRO" ]; then
    echo "ROS environment not sourced. Please run: source /opt/ros/noetic/setup.bash"
    MISSING_DEPS=1
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo "Some dependencies are missing. Please install them before proceeding."
else
    echo "All dependencies found."
fi

# Ensure proper CUDA device is detected
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "CUDA device information:"
    nvidia-smi --query-gpu=name,compute_cap --format=csv
else
    echo "Warning: nvidia-smi not found - cannot verify CUDA device"
fi

echo "Environment setup complete. You may need to source this script in your shell."
echo "  source $SCRIPT_DIR/setup_environment.sh"