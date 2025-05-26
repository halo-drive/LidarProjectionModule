#!/bin/bash

# Start development container with ROS networking to host


# Source host ROS setup to get ROS_MASTER_URI and other vars

source /opt/ros/noetic/setup.bash

# Ensure host.docker.internal resolves to host IP (for ROS communication)

DOCKER_HOST_IP=$(ip -4 addr show docker0 | grep -Po 'inet \K[\d.]+')
echo "$DOCKER_HOST_IP host.docker.internal" | sudo tee -a /etc/hosts

# Launch container directly with docker run

docker run --gpus all -it \
  -v $(pwd)/..:/workspace/LidarProjectionLane \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /opt/ros/noetic:/opt/ros/noetic:ro \
  -v ~/ws_vel:/workspace/ws_vel:ro \
  -v ~/pytorch_builds:/workspace/pytorch_builds:ro \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e ROS_MASTER_URI=http://host.docker.internal:11311 \
  -e ROS_HOSTNAME=LidarProjectionLane \
  --network host \
  --privileged \
  --workdir /workspace/LidarProjectionLane \
  lidarprojectiondev \
  bash -c "source /opt/ros/noetic/setup.bash && exec bash"
