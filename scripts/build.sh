#!/bin/bash
# Build lane_fusion package within ROS Catkin workspace

# Source ROS
source /opt/ros/noetic/setup.bash

# Create catkin workspace if not exists
mkdir -p /workspace/catkin_ws/src
ln -sf /workspace/LidarProjectionLane /workspace/catkin_ws/src/

# Build with catkin
cd /workspace/catkin_ws
catkin init
catkin build
source devel/setup.bash

echo "Build complete. Source workspace with: source /workspace/catkin_ws/devel/setup.bash"