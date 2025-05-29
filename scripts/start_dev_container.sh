#!/bin/bash

# Persistent TensorRT Development Container Script
# Container persists across sessions

CONTAINER_NAME="lidarprojection-dev"
IMAGE_NAME="lidarprojectiondev"

echo "Managing TensorRT Development Container..."

# Add host.docker.internal entry (run once)
echo "172.17.0.1 host.docker.internal" | sudo tee -a /etc/hosts 2>/dev/null || true

# Check if container already exists
if docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' exists."

    # Check if it's running
    if docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container is already running. Attaching..."
        docker exec -it ${CONTAINER_NAME} bash
    else
        echo "Starting existing container..."
        docker start ${CONTAINER_NAME}
        docker exec -it ${CONTAINER_NAME} bash
    fi
else
    echo "Creating new persistent container..."
    docker create --gpus all -it \
      --name ${CONTAINER_NAME} \
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
      ${IMAGE_NAME} \
      bash

    echo "Starting container and entering shell..."
    docker start ${CONTAINER_NAME}
    docker exec -it ${CONTAINER_NAME} bash -c "
        echo '=== TensorRT Environment Setup ===' && \
        source /opt/ros/noetic/setup.bash && \
        echo 'CUDA Version:' && nvcc --version && \
        echo 'TensorRT Version:' && cat /usr/src/tensorrt/version.txt 2>/dev/null || echo 'TensorRT found in /usr/src/tensorrt' && \
        echo 'GPU Status:' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && \
        echo '=== Ready for Development ===' && \
        cd /workspace/ws_LPM && \
        exec bash
    "
fi