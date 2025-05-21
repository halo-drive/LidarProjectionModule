cmake_minimum_required(VERSION 3.10)
project(lane_fusion)

# Find required packages
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        sensor_msgs
        cv_bridge
        pcl_ros
        velodyne_msgs
)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

# Optional: Find TensorRT
find_package(TensorRT)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_87") # Jetson AGX Orin

# Declare catkin package
catkin_package(
        INCLUDE_DIRS include
        CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs cv_bridge pcl_ros velodyne_msgs
        DEPENDS OpenCV PCL CUDA Eigen3
)

# Include directories
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
)

# Add rest of build instructions here
# ...