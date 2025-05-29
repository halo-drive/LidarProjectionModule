#!/bin/bash
# LidarProjectionLane Environment Setup Script
# Configures complete build environment for direct catkin_make usage
# Technical implementation for embedded systems development

set -e

echo "==========================================="
echo "LidarProjectionLane Environment Setup"
echo "==========================================="

# Directory resolution
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"

echo "Project structure:"
echo "  Workspace: $WORKSPACE_ROOT"
echo "  Project:   $PROJECT_ROOT"

# =======================
# Core Environment Setup
# =======================

# Project-specific paths
export LANE_FUSION_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT/scripts"

# =======================
# CUDA Configuration
# =======================

# Auto-detect CUDA installation
CUDA_PATHS=("/usr/local/cuda" "/opt/cuda" "/usr/local/cuda-11.4" "/usr/local/cuda-11.8")
CUDA_FOUND=false

for cuda_path in "${CUDA_PATHS[@]}"; do
    if [ -d "$cuda_path" ] && [ -f "$cuda_path/bin/nvcc" ]; then
        export CUDA_HOME="$cuda_path"
        export CUDA_ROOT="$cuda_path"
        export CUDA_TOOLKIT_ROOT_DIR="$cuda_path"
        export PATH="$cuda_path/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_path/lib64:$cuda_path/lib:$LD_LIBRARY_PATH"
        CUDA_FOUND=true
        echo "CUDA found: $cuda_path"
        break
    fi
done

if [ "$CUDA_FOUND" = false ]; then
    echo "ERROR: CUDA installation not found in standard locations"
    exit 1
fi

# Verify CUDA version compatibility
if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "CUDA version: $CUDA_VERSION"
fi

# =======================
# TensorRT Configuration
# =======================

# Auto-detect TensorRT installation
TENSORRT_PATHS=("/workspace/TensorRT-8.5.2.2" "/usr/src/tensorrt" "/opt/tensorrt" "/usr/local/tensorrt")
TENSORRT_FOUND=false

for trt_path in "${TENSORRT_PATHS[@]}"; do
    if [ -d "$trt_path" ] && [ -f "$trt_path/include/NvInfer.h" ]; then
        export TENSORRT_ROOT="$trt_path"
        export TensorRT_ROOT="$trt_path"
        export TENSORRT_INCLUDE_DIR="$trt_path/include"
        export TENSORRT_LIBRARY_DIR="$trt_path/lib"
        export LD_LIBRARY_PATH="$trt_path/lib:$LD_LIBRARY_PATH"
        TENSORRT_FOUND=true
        echo "TensorRT found: $trt_path"
        break
    fi
done

if [ "$TENSORRT_FOUND" = false ]; then
    echo "ERROR: TensorRT installation not found"
    echo "Expected locations: ${TENSORRT_PATHS[*]}"
    exit 1
fi

# =======================
# cuDNN Configuration
# =======================

# Auto-detect cuDNN installation
CUDNN_PATHS=("$CUDA_HOME" "/usr/local/cudnn" "/opt/cudnn")
CUDNN_FOUND=false

for cudnn_path in "${CUDNN_PATHS[@]}"; do
    if [ -f "$cudnn_path/include/cudnn.h" ] || [ -f "$cudnn_path/include/cudnn_version.h" ]; then
        export CUDNN_ROOT_DIR="$cudnn_path"
        export LD_LIBRARY_PATH="$cudnn_path/lib64:$cudnn_path/lib:$LD_LIBRARY_PATH"
        CUDNN_FOUND=true
        echo "cuDNN found: $cudnn_path"
        break
    fi
done

if [ "$CUDNN_FOUND" = false ]; then
    echo "WARNING: cuDNN not found - will proceed without it"
    echo "For optimal performance, consider installing cuDNN 8.x for CUDA 11.8"
fi

# =======================
# CMake Environment Variables
# =======================

# Set CMake cache variables as environment variables
export CMAKE_PREFIX_PATH="$TENSORRT_ROOT:$CUDA_HOME:$CMAKE_PREFIX_PATH"

# Specific CMake variables for dependency discovery
export TensorRT_DIR="$TENSORRT_ROOT"
if [ "$CUDNN_FOUND" = true ]; then
    export CUDNN_ROOT_DIR="$CUDNN_ROOT_DIR"
fi

# Build configuration
export CMAKE_BUILD_TYPE="Release"

# =======================
# GPU Architecture Detection
# =======================

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU device information:"
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1

    # Auto-detect GPU architecture for CMAKE_CUDA_ARCHITECTURES
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d '.')
    if [ ! -z "$GPU_ARCH" ]; then
        export CMAKE_CUDA_ARCHITECTURES="$GPU_ARCH"
        echo "Auto-detected GPU architecture: $GPU_ARCH"
    fi
else
    echo "WARNING: nvidia-smi not available - using default GPU architecture"
    export CMAKE_CUDA_ARCHITECTURES="86"
fi

# =======================
# ROS Environment
# =======================

if [ -z "$ROS_DISTRO" ]; then
    if [ -f "/opt/ros/noetic/setup.bash" ]; then
        echo "Sourcing ROS Noetic environment"
        source /opt/ros/noetic/setup.bash
        export ROS_DISTRO="noetic"
    else
        echo "ERROR: ROS installation not found"
        exit 1
    fi
fi

echo "ROS environment: $ROS_DISTRO"

# =======================
# Dependency Verification
# =======================

echo "Verifying build dependencies..."

MISSING_DEPS=0
check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "MISSING: $1"
        MISSING_DEPS=1
    else
        echo "FOUND: $1"
    fi
}

check_command cmake
check_command nvcc
check_command python3
check_command catkin_make

# Verify critical libraries
verify_library() {
    local lib_name="$1"
    local search_paths="$2"

    for path in $search_paths; do
        if [ -f "$path/lib$lib_name.so" ] || [ -f "$path/lib64/lib$lib_name.so" ]; then
            echo "FOUND: lib$lib_name"
            return 0
        fi
    done
    echo "MISSING: lib$lib_name"
    return 1
}

# Critical TensorRT libraries
TRT_LIB_PATHS="$TENSORRT_ROOT/lib $TENSORRT_ROOT/lib64"
verify_library "nvinfer" "$TRT_LIB_PATHS" || MISSING_DEPS=1
verify_library "nvonnxparser" "$TRT_LIB_PATHS" || MISSING_DEPS=1

# =======================
# Project-Specific Checks
# =======================

# Verify ONNX model
ONNX_MODEL="$PROJECT_ROOT/lane_detection/models/yolov8n-seg-lane.onnx"
if [ -f "$ONNX_MODEL" ]; then
    echo "FOUND: ONNX model $(basename "$ONNX_MODEL")"
    MODEL_SIZE=$(ls -lh "$ONNX_MODEL" | awk '{print $5}')
    echo "Model size: $MODEL_SIZE"
else
    echo "MISSING: ONNX model at $ONNX_MODEL"
    MISSING_DEPS=1
fi

# Verify catkin workspace structure
if [ ! -f "$WORKSPACE_ROOT/CMakeLists.txt" ]; then
    echo "ERROR: Invalid catkin workspace structure"
    echo "Expected: $WORKSPACE_ROOT/src/CMakeLists.txt"
    MISSING_DEPS=1
else
    echo "FOUND: Valid catkin workspace structure"
fi

# =======================
# Final Status Report
# =======================

echo ""
echo "==========================================="
if [ $MISSING_DEPS -eq 1 ]; then
    echo "ENVIRONMENT SETUP INCOMPLETE"
    echo "Resolve missing dependencies before building"
    exit 1
else
    echo "ENVIRONMENT SETUP COMPLETE"
fi
echo "==========================================="

echo "Configuration summary:"
echo "  CUDA:          $CUDA_HOME"
echo "  TensorRT:      $TENSORRT_ROOT"
echo "  cuDNN:         ${CUDNN_ROOT_DIR:-Not found}"
echo "  ROS:           $ROS_DISTRO"
echo "  GPU Arch:      $CMAKE_CUDA_ARCHITECTURES"
echo "  Workspace:     $WORKSPACE_ROOT"
echo ""
echo "Build procedure:"
echo "  cd $WORKSPACE_ROOT"
echo "  catkin_make"
echo ""
echo "NOTE: First build will convert ONNX to TensorRT engine (5-10 minutes)"
echo "Subsequent builds will use cached engine for faster execution"
echo "==========================================="