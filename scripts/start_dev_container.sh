#!/bin/bash

# Fixed TensorRT Development Container Script
# TensorRT 8.5.2.2 with full GPU support

echo "Starting TensorRT Development Container..."

# Add host.docker.internal entry
echo "172.17.0.1 host.docker.internal" | sudo tee -a /etc/hosts

docker run --gpus all -it --rm \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /opt/ros/noetic:/opt/ros/noetic:ro \
  -v ~/ws_vel:/workspace/ws_vel:ro \
  -v ~/ws_LPM:/workspace/ws_LPM \
  -v ~/pytorch_builds:/workspace/pytorch_builds \
  -v /dev:/dev \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e ROS_MASTER_URI=http://host.docker.internal:11311 \
  -e ROS_HOSTNAME=LidarProjectionLane \
  -e TENSORRT_ROOT=/usr/src/tensorrt \
  -e CUDA_ROOT=/usr/local/cuda \
  -e LD_LIBRARY_PATH=/usr/src/tensorrt/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
  --network host \
  --privileged \
  --shm-size=8g \
  --workdir /workspace/ws_LPM \
  lidarprojectiondev \
  bash -c "
    echo '=== TensorRT Environment Setup ===' && \
    source /opt/ros/noetic/setup.bash && \
    echo 'CUDA Version:' && nvcc --version && \
    echo 'TensorRT Version:' && cat /usr/src/tensorrt/version.txt 2>/dev/null || echo 'TensorRT found in /usr/src/tensorrt' && \
    echo 'GPU Status:' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && \
    echo '=== Ready for Development ===' && \
    cd /workspace/ws_LPM && \
    exec bash
  "