#!/bin/bash
# LidarProjectionLane System Status Checker

echo "========================================="
echo "  LidarProjectionLane System Status"
echo "========================================="

# Check project structure
echo -e "\n📁 Project Structure:"
if [ -d "/workspace/ws_LPM/src/LidarProjectionLane" ]; then
    echo "✓ Main project found: /workspace/ws_LPM/src/LidarProjectionLane"
    echo "  Modules present:"
    for dir in calibration lane_detection lidar_processing fusion utils; do
        if [ -d "/workspace/ws_LPM/src/LidarProjectionLane/$dir" ]; then
            echo "  ✓ $dir"
        else
            echo "  ✗ $dir (missing)"
        fi
    done
else
    echo "✗ Main project not found"
fi

# Check build status
echo -e "\n🔨 Build Status:"
if [ -d "/workspace/ws_LPM/build" ]; then
    echo "✓ Catkin build directory exists"
    if [ -f "/workspace/ws_LPM/devel/setup.bash" ]; then
        echo "✓ Project has been built (devel/setup.bash exists)"
    else
        echo "⚠ Project may not be built yet"
    fi
else
    echo "✗ No build directory found"
fi

# Check PyTorch
echo -e "\n🔥 PyTorch Status:"
if python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} installed')" 2>/dev/null; then
    python3 -c "
import torch
cuda_available = torch.cuda.is_available()
print(f'  CUDA Available: {\"✓\" if cuda_available else \"✗\"} {cuda_available}')
if cuda_available:
    print(f'  CUDA Version: {torch.version.cuda}')
    print(f'  GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
else
    echo "✗ PyTorch not installed or not working"

    # Check for pre-built wheels
    if [ -f "/workspace/pytorch_builds/wheels/torch-2.1.0"*".whl" ]; then
        echo "  ℹ Pre-built PyTorch wheel found in /workspace/pytorch_builds/wheels/"
        echo "  → Run: ./install_pytorch.sh"
    else
        echo "  ℹ No pre-built wheel found"
        echo "  → Run: ./build_pytorch.sh (1-3 hours) or install via pip"
    fi
fi

# Check CUDA system
echo -e "\n⚡ CUDA System:"
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA driver installed"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
else
    echo "✗ NVIDIA driver not found"
fi

if [ -d "/usr/local/cuda" ]; then
    cuda_version=$(cat /usr/local/cuda/version.txt 2>/dev/null || echo "Version file not found")
    echo "✓ CUDA toolkit: $cuda_version"
else
    echo "✗ CUDA toolkit not found"
fi

# Check ROS
echo -e "\n🤖 ROS Status:"
if [ -f "/opt/ros/noetic/setup.bash" ]; then
    echo "✓ ROS Noetic installed"
    if [ -n "$ROS_DISTRO" ]; then
        echo "  ✓ ROS environment active: $ROS_DISTRO"
    else
        echo "  ⚠ ROS environment not sourced"
        echo "  → Run: source /opt/ros/noetic/setup.bash"
    fi
else
    echo "✗ ROS Noetic not found"
fi

# Check dependencies
echo -e "\n📦 Key Dependencies:"
deps=("opencv-python" "ultralytics" "numpy" "scipy")
for dep in "${deps[@]}"; do
    if python3 -c "import ${dep//-/_}" 2>/dev/null; then
        version=$(python3 -c "import ${dep//-/_}; print(getattr(${dep//-/_}, '__version__', 'unknown'))" 2>/dev/null)
        echo "  ✓ $dep ($version)"
    else
        echo "  ✗ $dep (not installed)"
    fi
done

# Check disk space
echo -e "\n💾 Storage:"
echo "Workspace usage:"
du -sh /workspace/* 2>/dev/null | sort -hr

echo -e "\nFree space:"
df -h /workspace | grep -v "Filesystem"

# Recommendations
echo -e "\n💡 Next Steps:"
if ! python3 -c "import torch" 2>/dev/null; then
    echo "1. Install PyTorch: Run ./install_pytorch.sh or ./build_pytorch.sh"
fi

if [ ! -f "/workspace/ws_LPM/devel/setup.bash" ]; then
    echo "2. Build project: cd /workspace/ws_LPM && catkin_make"
fi

if [ -z "$ROS_DISTRO" ]; then
    echo "3. Source ROS: source /opt/ros/noetic/setup.bash"
fi

echo "4. Source workspace: source /workspace/ws_LPM/devel/setup.bash"
echo "5. Launch system: roslaunch lane_fusion sensors.launch"

echo -e "\n========================================="