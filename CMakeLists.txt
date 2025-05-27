cmake_minimum_required(VERSION 3.10)
project(lane_fusion)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language
enable_language(CUDA)

# Find required ROS packages
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        sensor_msgs
        geometry_msgs
        cv_bridge
        image_transport
        pcl_ros
        velodyne_msgs
        message_filters
        tf2
        tf2_ros
        tf2_geometry_msgs
)

# Find system dependencies
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io features search kdtree)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

# Optional: Find TensorRT (for Phase 3)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY nvinfer
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
    set(TENSORRT_FOUND TRUE)
    message(STATUS "Found TensorRT: ${TENSORRT_LIBRARY}")
else()
    set(TENSORRT_FOUND FALSE)
    message(WARNING "TensorRT not found - will be needed for Phase 3")
endif()

# CUDA compilation flags for x86 platform
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3; -gencode arch=compute_75,code=sm_75)

# Declare catkin package (minimal for Phase 1-2)
catkin_package(
        CATKIN_DEPENDS
        roscpp
        rospy
        std_msgs
        sensor_msgs
        geometry_msgs
        cv_bridge
        image_transport
        pcl_ros
        velodyne_msgs
        message_filters
        tf2
        tf2_ros
        tf2_geometry_msgs
        DEPENDS
        OpenCV
        PCL
        CUDA
)

# Include directories (only existing ones)
include_directories(
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
)

if(TENSORRT_FOUND)
    include_directories(${TENSORRT_INCLUDE_DIR})
endif()

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Phase 3+ will add subdirectories and libraries here
# For now, just ensure the build system works

# Install launch files (corrected syntax)
install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
        FILES_MATCHING PATTERN "*.launch" PATTERN "*.py"
        PATTERN ".svn" EXCLUDE)

# Install config files
install(DIRECTORY config/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
        PATTERN ".svn" EXCLUDE)

# Install scripts
install(PROGRAMS
        scripts/setup_environment.sh
        scripts/build.sh
        scripts/calibration_collection.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Testing (optional)
if(CATKIN_ENABLE_TESTING)
    find_package(rostest REQUIRED)
    # Phase 3+ will add tests here
endif()

# Build status message
message(STATUS "=== LidarProjectionLane Build Configuration ===")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "PCL: Found")
message(STATUS "CUDA: ${CUDA_VERSION}")
message(STATUS "Eigen3: ${EIGEN3_VERSION}")
if(TENSORRT_FOUND)
    message(STATUS "TensorRT: Found")
else()
    message(STATUS "TensorRT: Not found (needed for Phase 3)")
endif()
message(STATUS "==============================================")