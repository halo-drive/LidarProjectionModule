#!/bin/bash

# Complete TensorRT Lane Detection Build Script
# For use inside Docker container with TensorRT 8.5.2.2

set -e  # Exit on any error

echo "========================================="
echo "TensorRT Lane Detection Build"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the workspace
if [ ! -f "src/LidarProjectionLane/CMakeLists.txt" ]; then
    echo -e "${RED}Error: Run this script from workspace root (/workspace/ws_LPM/)${NC}"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo -e "${GREEN}✓ Found workspace structure${NC}"

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"

# Check CUDA
if ! command -v nvcc >/dev/null 2>&1; then
    echo -e "${RED}Error: CUDA not found${NC}"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
echo -e "${GREEN}✓ CUDA found: $CUDA_VERSION${NC}"

# Check TensorRT
if [ ! -d "$TENSORRT_ROOT" ]; then
    echo -e "${RED}Error: TensorRT not found at $TENSORRT_ROOT${NC}"
    exit 1
fi

echo -e "${GREEN}✓ TensorRT found: $TENSORRT_ROOT${NC}"

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
echo -e "${GREEN}✓ GPU detected${NC}"

# Check OpenCV
if ! pkg-config --exists opencv4; then
    echo -e "${RED}Error: OpenCV 4 not found${NC}"
    echo "Installing OpenCV..."
    apt update && apt install -y libopencv-dev
fi

OPENCV_VERSION=$(pkg-config --modversion opencv4)
echo -e "${GREEN}✓ OpenCV found: $OPENCV_VERSION${NC}"

# Check ROS
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${YELLOW}Sourcing ROS...${NC}"
    source /opt/ros/noetic/setup.bash
fi

echo -e "${GREEN}✓ ROS found: $ROS_DISTRO${NC}"

# Install additional dependencies
echo -e "${YELLOW}Installing additional dependencies...${NC}"
apt update -qq
apt install -y \
    ros-$ROS_DISTRO-cv-bridge \
    ros-$ROS_DISTRO-image-transport \
    ros-$ROS_DISTRO-pcl-ros \
    libyaml-cpp-dev \
    libgtest-dev

# Verify ONNX model exists
ONNX_MODEL="src/LidarProjectionLane/lane_detection/models/yolov8n-seg-lane.onnx"
if [ ! -f "$ONNX_MODEL" ]; then
    echo -e "${RED}Error: ONNX model not found: $ONNX_MODEL${NC}"
    echo "Please ensure the YOLOv8n-seg-lane.onnx model is in the correct location"
    exit 1
fi

echo -e "${GREEN}✓ Found ONNX model: $ONNX_MODEL${NC}"
ls -lh "$ONNX_MODEL"

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p src/LidarProjectionLane/lane_detection/cuda
mkdir -p src/LidarProjectionLane/config/rviz

# Clean previous build
echo -e "${YELLOW}Cleaning previous build...${NC}"
rm -rf build/ devel/

# Source ROS
source /opt/ros/noetic/setup.bash

# Set build environment
export CMAKE_BUILD_TYPE=Release
export CUDA_ARCHITECTURES=75  # Adjust for your GPU: RTX 20xx=75, RTX 30xx=86, RTX 40xx=89

# Determine GPU architecture automatically
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d '.')
if [ ! -z "$GPU_ARCH" ]; then
    export CUDA_ARCHITECTURES=$GPU_ARCH
    echo -e "${BLUE}Auto-detected GPU architecture: $GPU_ARCH${NC}"
fi

# Build the workspace
echo -e "${YELLOW}Building workspace with TensorRT support...${NC}"
echo -e "${BLUE}This may take 10-15 minutes for first build...${NC}"

catkin_make -DCMAKE_BUILD_TYPE=Release \
           -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
           -DTENSORRT_ROOT="$TENSORRT_ROOT" \
           -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
           -DCUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
           -j$(nproc)

# Check build result
if [ $? -eq 0 ]; then
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}✓ TensorRT Build Successful!${NC}"
    echo -e "${GREEN}=========================================${NC}"

    # Source the workspace
    source devel/setup.bash

    # Verify executables
    if [ -f "devel/lib/lane_fusion/lane_detection_node" ]; then
        echo -e "${GREEN}✓ lane_detection_node created${NC}"
    else
        echo -e "${RED}✗ lane_detection_node NOT found${NC}"
        exit 1
    fi

    # Show build artifacts
    echo -e "${YELLOW}Build artifacts:${NC}"
    ls -la devel/lib/lane_fusion/

    echo ""
    echo -e "${BLUE}=== Next Steps ===${NC}"
    echo "1. Source the workspace:"
    echo "   source devel/setup.bash"
    echo ""
    echo "2. Test TensorRT conversion (first run will be slow):"
    echo "   roscore &"
    echo "   # Wait a moment, then:"
    echo "   rosrun lane_fusion lane_detection_node"
    echo ""
    echo "3. Full system test:"
    echo "   # Terminal 1:"
    echo "   roslaunch lane_fusion sensors.launch"
    echo "   # Terminal 2:"
    echo "   roslaunch lane_fusion lane_detection.launch"
    echo ""
    echo "4. Monitor performance:"
    echo "   rostopic echo /camera0/lane_statistics"
    echo "   nvidia-smi -l 1"
    echo ""
    echo -e "${GREEN}Ready for TensorRT lane detection!${NC}"

    # Performance info
    echo ""
    echo -e "${BLUE}=== Expected Performance ===${NC}"
    echo "• First run: 30-60s (ONNX → TensorRT conversion)"
    echo "• Subsequent runs: <2s startup"
    echo "• Inference: 8-15ms per frame"
    echo "• Overall FPS: 25-30 Hz"
    echo "• GPU Memory: 2-3GB"

else
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}✗ Build Failed!${NC}"
    echo -e "${RED}=========================================${NC}"
    echo ""
    echo -e "${YELLOW}Common solutions:${NC}"
    echo "1. Check CUDA/TensorRT paths:"
    echo "   ls -la /usr/src/tensorrt/lib/"
    echo "   ls -la /usr/local/cuda/lib64/"
    echo ""
    echo "2. Verify GPU architecture:"
    echo "   nvidia-smi --query-gpu=compute_cap --format=csv"
    echo ""
    echo "3. Check available memory:"
    echo "   nvidia-smi"
    echo "   free -h"
    echo ""
    echo "4. Review build log above for specific errors"
    exit 1
fi