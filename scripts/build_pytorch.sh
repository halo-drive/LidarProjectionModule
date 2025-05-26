#!/bin/bash
# Build PyTorch 2.1.0 with CUDA 11.4 support
set -e

BUILD_DIR="/workspace/pytorch_builds/cuda11.4_torch2.1.0"
WHEELS_DIR="/workspace/pytorch_builds/wheels"

echo "Building PyTorch 2.1.0 with CUDA 11.4 support..."
echo "Build directory: $BUILD_DIR"
echo "Wheels directory: $WHEELS_DIR"

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$WHEELS_DIR"

# Check if already built
if [ -f "$WHEELS_DIR/torch-2.1.0-*.whl" ]; then
    echo "PyTorch wheel already exists. Installing..."
    pip3 install "$WHEELS_DIR"/torch-2.1.0-*.whl --force-reinstall
    exit 0
fi

# Install build dependencies
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    ninja-build \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    ccache

# Install Python build dependencies
pip3 install numpy pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

# Set environment variables for CUDA 11.4
export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# PyTorch build configuration
export USE_CUDA=1
export CUDA_ARCH_LIST="8.6"  # RTX 3060
export TORCH_CUDA_ARCH_LIST="8.6"
export FORCE_CUDA=1
export USE_CUDNN=1
export USE_MKLDNN=1
export USE_OPENMP=1
export USE_LAPACK=1
export BUILD_TEST=0
export MAX_JOBS=4

# Setup ccache
export CC="ccache gcc"
export CXX="ccache g++"

# Clone PyTorch if not exists
cd "$BUILD_DIR"
if [ ! -d "pytorch" ]; then
    echo "Cloning PyTorch..."
    git clone --recursive https://github.com/pytorch/pytorch.git
fi

cd pytorch

# Checkout 2.1.0 and update submodules
echo "Checking out PyTorch 2.1.0..."
git checkout v2.1.0
git submodule sync
git submodule update --init --recursive

# Clean previous builds
python3 setup.py clean

echo "Starting PyTorch build (this will take 1-3 hours)..."
echo "Progress will be saved to $BUILD_DIR/build.log"

# Build with output logging
python3 setup.py bdist_wheel 2>&1 | tee "$BUILD_DIR/build.log"

# Copy wheel to persistent location
cp dist/torch-2.1.0-*.whl "$WHEELS_DIR/"

echo "Build complete! Installing wheel..."
pip3 install "$WHEELS_DIR"/torch-2.1.0-*.whl

echo "PyTorch 2.1.0 with CUDA 11.4 successfully built and installed!"