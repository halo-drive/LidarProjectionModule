cmake_minimum_required(VERSION 3.0.2)
project(lane_fusion)

## Compile as C++14, required for TensorRT
add_compile_options(-std=c++14)

##############################################
## Find catkin macros and libraries
##############################################
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        sensor_msgs
        geometry_msgs
        visualization_msgs
        cv_bridge
        image_transport
        pcl_ros
        pcl_conversions
        velodyne_msgs
        message_filters
        tf2
        tf2_ros
        tf2_geometry_msgs
        tf2_eigen
)

##############################################
## System dependencies
##############################################
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io features search kdtree segmentation filters)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

# CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

## CUDA architecture settings
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)  # RTX 20xx series - adjust as needed
endif()

# Define CUDA compilation flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES})

##############################################
## TensorRT - Multiple search paths for different installations
##############################################
set(TENSORRT_ROOT "/workspace/TensorRT-8.5.2.2" CACHE PATH "TensorRT root directory")

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_PLUGIN nvinfer_plugin
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)

# Check if TensorRT was found
if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY_INFER AND TENSORRT_LIBRARY_ONNXPARSER)
    set(TENSORRT_FOUND TRUE)
    message(STATUS "Found TensorRT:")
    message(STATUS "  Include: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "  Libraries: ${TENSORRT_LIBRARY_INFER}")
else()
    set(TENSORRT_FOUND FALSE)
    message(WARNING "TensorRT not found - will be needed for lane detection")
endif()

##############################################
## catkin specific configuration
##############################################
catkin_package(
        INCLUDE_DIRS 
            lidar_processing/include
            lane_detection/include
        LIBRARIES 
            ${PROJECT_NAME}_lidar_processing
            ${PROJECT_NAME}_tensor_utils
            ${PROJECT_NAME}_yolo_detector
            ${PROJECT_NAME}_lane_segmentation
        CATKIN_DEPENDS
            roscpp
            rospy
            std_msgs
            sensor_msgs
            geometry_msgs
            visualization_msgs
            cv_bridge
            image_transport
            pcl_ros
            pcl_conversions
            velodyne_msgs
            message_filters
            tf2
            tf2_ros
            tf2_geometry_msgs
            tf2_eigen
        DEPENDS
            OpenCV
            PCL
            Boost
)

##############################################
## Build Configuration
##############################################

## Specify additional locations of header files
include_directories(
        lidar_processing/include
        lane_detection/include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)

if(TENSORRT_FOUND)
    include_directories(${TENSORRT_INCLUDE_DIR})
endif()

# Add PCL definitions
add_definitions(${PCL_DEFINITIONS})

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

##############################################
## Libraries
##############################################

###################
## LiDAR Processing Libraries (YOUR WORK)
###################

# Point cloud processing library
add_library(${PROJECT_NAME}_lidar_processing
    lidar_processing/src/point_cloud_proc.cpp
    lidar_processing/src/ground_extraction.cpp
)

target_link_libraries(${PROJECT_NAME}_lidar_processing
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
)

# CUDA ground extraction library
if(CUDA_FOUND)
    cuda_add_library(${PROJECT_NAME}_cuda_ground_extraction
        lidar_processing/cuda/ground_extraction.cu
    )
    
    target_link_libraries(${PROJECT_NAME}_cuda_ground_extraction
        ${CUDA_LIBRARIES}
    )
    
    # Link CUDA library to main library
    target_link_libraries(${PROJECT_NAME}_lidar_processing
        ${PROJECT_NAME}_cuda_ground_extraction
    )
endif()

###################
## Lane Detection Libraries (TENSORRT WORK)
###################

if(TENSORRT_FOUND)
    # CUDA Tensor Utilities
    cuda_add_library(${PROJECT_NAME}_tensor_utils
            lane_detection/src/tensor_utils.cpp
    )

    target_link_libraries(${PROJECT_NAME}_tensor_utils
            ${catkin_LIBRARIES}
            ${OpenCV_LIBRARIES}
            ${CUDA_LIBRARIES}
            ${TENSORRT_LIBRARY_INFER}
            ${TENSORRT_LIBRARY_ONNXPARSER}
            ${TENSORRT_LIBRARY_PLUGIN}
            cublas
            curand

    )

    # YOLO Detector Library with TensorRT
    add_library(${PROJECT_NAME}_yolo_detector
            lane_detection/src/yolo_detector.cpp
    )

    target_link_libraries(${PROJECT_NAME}_yolo_detector
            ${PROJECT_NAME}_tensor_utils
            ${catkin_LIBRARIES}
            ${OpenCV_LIBRARIES}
            ${CUDA_LIBRARIES}
            ${TENSORRT_LIBRARY_INFER}
            ${TENSORRT_LIBRARY_ONNXPARSER}
            ${TENSORRT_LIBRARY_PLUGIN}
    )

    # Lane Segmentation Library
    add_library(${PROJECT_NAME}_lane_segmentation
            lane_detection/src/lane_segmentation.cpp
    )

    target_link_libraries(${PROJECT_NAME}_lane_segmentation
            ${catkin_LIBRARIES}
            ${OpenCV_LIBRARIES}
    )
endif()

##############################################
## Executables
##############################################

# Ground extraction node executable (YOUR WORK)
add_executable(lidar_ground_extraction_node
    lidar_processing/src/lidar_ground_extraction_node.cpp
)

target_link_libraries(lidar_ground_extraction_node
    ${PROJECT_NAME}_lidar_processing
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
)

# Lane Detection Node (TENSORRT WORK)
if(TENSORRT_FOUND)
    add_executable(lane_detection_node
            lane_detection/src/lane_detection_node.cpp
    )

    target_link_libraries(lane_detection_node
            ${PROJECT_NAME}_yolo_detector
            ${PROJECT_NAME}_lane_segmentation
            ${PROJECT_NAME}_tensor_utils
            ${catkin_LIBRARIES}
            ${OpenCV_LIBRARIES}
    )
endif()

##############################################
## Dependencies
##############################################

# Add dependencies
add_dependencies(${PROJECT_NAME}_lidar_processing 
    ${${PROJECT_NAME}_EXPORTED_TARGETS} 
    ${catkin_EXPORTED_TARGETS}
)

add_dependencies(lidar_ground_extraction_node 
    ${${PROJECT_NAME}_EXPORTED_TARGETS} 
    ${catkin_EXPORTED_TARGETS}
)

if(TENSORRT_FOUND)
    add_dependencies(${PROJECT_NAME}_tensor_utils 
        ${${PROJECT_NAME}_EXPORTED_TARGETS} 
        ${catkin_EXPORTED_TARGETS}
    )
    
    add_dependencies(lane_detection_node 
        ${${PROJECT_NAME}_EXPORTED_TARGETS} 
        ${catkin_EXPORTED_TARGETS}
    )
endif()

##############################################
## Install
##############################################

# Install executables
install(TARGETS 
        lidar_ground_extraction_node
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(TENSORRT_FOUND)
    install(TARGETS 
            lane_detection_node
            RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
endif()

# Install libraries
install(TARGETS 
        ${PROJECT_NAME}_lidar_processing
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

if(TENSORRT_FOUND)
    install(TARGETS
            ${PROJECT_NAME}_yolo_detector
            ${PROJECT_NAME}_lane_segmentation
            ${PROJECT_NAME}_tensor_utils
            ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
    )
endif()

# Install header files
install(DIRECTORY lidar_processing/include/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.hpp"
)

if(TENSORRT_FOUND)
    install(DIRECTORY lane_detection/include/
            DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}/lane_detection
            FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
endif()

# Install launch files
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

##############################################
## Build Status
##############################################

message(STATUS "=== Lane Fusion Build Configuration ===")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "PCL: Found")
message(STATUS "CUDA: ${CUDA_VERSION}")
message(STATUS "Eigen3: ${EIGEN3_VERSION}")
if(TENSORRT_FOUND)
    message(STATUS "TensorRT: Found - Lane detection enabled")
else()
    message(STATUS "TensorRT: Not found - Lane detection disabled")
endif()
message(STATUS "Build targets:")
message(STATUS "  - lidar_ground_extraction_node (LiDAR processing)")
if(TENSORRT_FOUND)
    message(STATUS "  - lane_detection_node (Lane detection)")
endif()
message(STATUS "==========================================")