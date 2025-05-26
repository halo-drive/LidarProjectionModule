#!/bin/bash
# Build script for LidarProjectionLane project
# Phase 1: Core Infrastructure Build System
# Optimized for x86 development platform with CUDA support

set -e

echo "=== LidarProjectionLane Build System ==="

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Check if we're in a ROS workspace
if [ ! -f "src/CMakeLists.txt" ] && [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Not in a catkin workspace or package directory"
    echo "Current directory: $(pwd)"
    echo "Expected: ROS catkin workspace or package with CMakeLists.txt"
    exit 1
fi

# Source ROS environment
echo "Sourcing ROS environment..."
if [ -f "/opt/ros/noetic/setup.bash" ]; then
    source /opt/ros/noetic/setup.bash
    echo "ROS Noetic sourced"
else
    echo "ROS Noetic not found"
    exit 1
fi

# Source any existing workspace
if [ -f "devel/setup.bash" ]; then
    source devel/setup.bash
    echo "Existing workspace sourced"
fi

# Check for CUDA
if ! command -v nvcc >/dev/null 2>&1; then
    echo "CUDA not found. CUDA is required for this project."
    exit 1
else
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5 $6}')
    echo "CUDA found: $CUDA_VERSION"
fi

# Set build configuration
BUILD_TYPE=${1:-Release}
PARALLEL_JOBS=${2:-$(nproc)}

echo "Build configuration:"
echo "  Build type: $BUILD_TYPE"
echo "  Parallel jobs: $PARALLEL_JOBS"
echo "  Project root: $PROJECT_ROOT"

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
    echo "Created build directory"
fi

# Clean build if requested
if [ "$3" = "clean" ]; then
    echo "Cleaning previous build..."
    rm -rf build/* devel/*
    echo "Build directories cleaned"
fi

# Check dependencies before building
echo ""
echo "=== Pre-build Dependency Check ==="
DEPS_OK=true

# Check for required tools
if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake not found"
    DEPS_OK=false
fi

if ! command -v make >/dev/null 2>&1; then
    echo "make not found"
    DEPS_OK=false
fi

# Check for required ROS packages
ROS_PACKAGES=("roscpp" "sensor_msgs" "cv_bridge" "pcl_ros")
for pkg in "${ROS_PACKAGES[@]}"; do
    if ! rospack find "$pkg" >/dev/null 2>&1; then
        echo "ROS package $pkg not found"
        DEPS_OK=false
    fi
done

if [ "$DEPS_OK" = false ]; then
    echo "Dependencies missing. Please install required packages."
    exit 1
fi

echo "All dependencies satisfied"

# Build the project
echo ""
echo "=== Starting Build Process ==="
echo "Timestamp: $(date)"

# Use catkin_make for ROS1
echo "Building with catkin_make..."

# Set CUDA architecture for x86 (adjust based on your GPU)
export CUDAARCHS="75"  # For RTX 20xx/30xx series, adjust as needed

# Build with optimizations
catkin_make \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_FLAGS="-arch=sm_75 -O3" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native" \
    -j$PARALLEL_JOBS

BUILD_EXIT_CODE=$?

echo ""
if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "✅ Build completed successfully!"

    # Source the workspace
    echo "Sourcing workspace..."
    source devel/setup.bash

    # Show build summary
    echo ""
    echo "=== Build Summary ==="
    echo "Timestamp: $(date)"
    echo "Build type: $BUILD_TYPE"
    echo "CUDA support: YES"

    # Check if key targets were built
    echo ""
    echo "Available ROS nodes:"
    if [ -f "devel/lib/lane_fusion/yolo_detector_node" ]; then
        echo " yolo_detector_node"
    else
        echo " yolo_detector_node (Phase 3)"
    fi

    if [ -f "devel/lib/lane_fusion/lidar_processor_node" ]; then
        echo "lidar_processor_node"
    else
        echo "lidar_processor_node (Phase 4)"
    fi

    if [ -f "devel/lib/lane_fusion/lane_fusion_node" ]; then
        echo "lane_fusion_node"
    else
        echo "lane_fusion_node (Phase 6)"
    fi

    echo ""
    echo "Launch files available:"
    if [ -f "src/lane_fusion/launch/sensors.launch" ]; then
        echo "sensors.launch"
    fi

    echo ""
    echo "Next steps:"
    echo "  1. Test sensor setup: roslaunch lane_fusion sensors.launch"
    echo "  2. Verify camera topics: rostopic list | grep camera"
    echo "  3. Check LiDAR topics: rostopic list | grep velodyne"

else
    echo "❌ Build failed with exit code: $BUILD_EXIT_CODE"
    echo ""
    echo "Common issues:"
    echo "  - Missing dependencies (run scripts/setup_environment.sh)"
    echo "  - CUDA version mismatch"
    echo "  - Insufficient memory during compilation"
    echo ""
    echo "To clean and rebuild:"
    echo "  $0 Release $(nproc) clean"

    exit $BUILD_EXIT_CODE
fi

echo ""
echo "=== Build Process Complete ==="