#!/bin/bash
# Setup cross-compilation tools for ARM64 Jetson

# Install cross-compilation toolchain
apt-get update && apt-get install -y \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    binutils-aarch64-linux-gnu

# Create toolchain file for CMake
cat > /workspace/aarch64_toolchain.cmake << EOF
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# CUDA settings for Jetson
set(CMAKE_CUDA_ARCHITECTURES 87) # For Jetson AGX Orin
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
EOF

# Make script executable
chmod +x /workspace/aarch64_toolchain.cmake