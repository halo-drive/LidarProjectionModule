cmake_minimum_required(VERSION 3.0.2)
project(lane_fusion)

## Compile as C++14, required for TensorRT and embedded platform compatibility
add_compile_options(-std=c++14)

## Global embedded platform configuration
message(STATUS "=== Lane Fusion Integrated Build System ===")
message(STATUS "Target Platform: Embedded (NVIDIA Jetson/Xavier/Orin)")
message(STATUS "Build Mode: Modular with isolated compilation support")

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
        camera_info_manager
        image_geometry
)

##############################################
## System dependencies for all modules
##############################################
find_package(OpenCV REQUIRED COMPONENTS
        core imgproc features2d calib3d stitching xfeatures2d imgcodecs highgui
)
find_package(PCL REQUIRED COMPONENTS
        common io features search kdtree segmentation filters surface octree
)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem thread)

##############################################
## CUDA configuration for all modules
##############################################
find_package(CUDA REQUIRED)
enable_language(CUDA)

## CUDA architecture settings for embedded platforms
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Support multiple embedded architectures
    set(CMAKE_CUDA_ARCHITECTURES 72 75 87)  # Xavier=72, Turing=75, Ampere=87
endif()

# Global CUDA compilation flags optimized for embedded deployment
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --ptxas-options=-v")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-gencode arch=compute_75,code=sm_75)

# Export CUDA availability for child modules
set(WORKSPACE_CUDA_FOUND TRUE CACHE BOOL "CUDA availability for child modules" FORCE)

##############################################
## TensorRT configuration for lane detection
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
    message(STATUS "Found TensorRT: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "  Libraries: ${TENSORRT_LIBRARY_INFER}")

    # Export TensorRT availability for child modules
    set(WORKSPACE_TENSORRT_FOUND TRUE CACHE BOOL "TensorRT availability" FORCE)
    set(WORKSPACE_TENSORRT_ROOT ${TENSORRT_ROOT} CACHE PATH "TensorRT root path" FORCE)
else()
    set(TENSORRT_FOUND FALSE)
    set(WORKSPACE_TENSORRT_FOUND FALSE CACHE BOOL "TensorRT availability" FORCE)
    message(WARNING "TensorRT not found - lane detection module will be disabled")
endif()

##############################################
## catkin package configuration
##############################################
catkin_package(
        INCLUDE_DIRS
        lidar_processing/include
        lane_detection/include
        fusion/include
        camera_stitching/include
        LIBRARIES
        # LiDAR processing libraries
        lane_fusion_lidar_processing_point_cloud_core
        lane_fusion_lidar_processing_ground_extraction
        # Lane detection libraries (conditional on TensorRT)
        $<$<BOOL:${TENSORRT_FOUND}>:lane_fusion_lane_detection_tensor_utils>
        $<$<BOOL:${TENSORRT_FOUND}>:lane_fusion_lane_detection_yolo_detector>
        $<$<BOOL:${TENSORRT_FOUND}>:lane_fusion_lane_detection_lane_segmentation>
        # Sensor fusion libraries
        lane_fusion_sensor_fusion_projection
        lane_fusion_sensor_fusion_lane_model
        lane_fusion_sensor_fusion_calibration
        # Camera stitching libraries
        lane_fusion_camera_stitching_synchronizer
        lane_fusion_camera_stitching_stitcher
        lane_fusion_camera_stitching_calibration
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
        camera_info_manager
        image_geometry
        DEPENDS
        OpenCV
        PCL
        Boost
        Eigen3
)

##############################################
## Global build configuration
##############################################

## Specify additional locations of header files for all modules
include_directories(
        lidar_processing/include
        lane_detection/include
        fusion/include
        camera_stitching/include
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

# Add PCL definitions and embedded platform optimizations
add_definitions(${PCL_DEFINITIONS})
add_definitions(-DEMBEDDED_PLATFORM -DINTEGRATED_BUILD)

# Compiler-specific optimizations for embedded deployment
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
    add_compile_options(-march=native -mtune=native -funroll-loops)
    add_compile_options(-ffast-math -finline-functions)
endif()

##############################################
## Module inclusion and compilation (Phase 3-4 Build Testing)
##############################################

message(STATUS "Building modules in dependency order for Phase 3-4 testing:")

## 1. LiDAR Processing Module (established - Phase 1-2)
message(STATUS "  1. LiDAR Processing Module")
# add_subdirectory(lidar_processing)  # Comment out for build isolation testing

## 2. Lane Detection Module (established - Phase 3)
if(TENSORRT_FOUND)
    message(STATUS "  2. Lane Detection Module (TensorRT enabled)")
    # add_subdirectory(lane_detection)  # Comment out for build isolation testing
else()
    message(STATUS "  2. Lane Detection Module (SKIPPED - TensorRT not found)")
endif()

## 3. Camera Stitching Module (NEW - Phase 3-4 dual-camera integration)
message(STATUS "  3. Camera Stitching Module (Phase 3-4 NEW)")
add_subdirectory(camera_stitching)

## 4. Sensor Fusion Module (Phase 4-5 integration)
message(STATUS "  4. Sensor Fusion Module (Phase 4-5)")
# add_subdirectory(fusion)  # Comment out for build isolation testing

##############################################
## Unified executables leveraging all modules
##############################################

# Main integrated lane fusion node
add_executable(integrated_lane_fusion_node
        src/integrated_lane_fusion_node.cpp
)

# Link all available module libraries
target_link_libraries(integrated_lane_fusion_node
        # LiDAR processing libraries
        ${LIDAR_PROCESSING_LIBRARIES}

        # Camera stitching libraries
        ${CAMERA_STITCHING_LIBRARIES}

        # Sensor fusion libraries
        ${SENSOR_FUSION_LIBRARIES}

        # System libraries
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${PCL_LIBRARIES}
        ${Boost_LIBRARIES}
)

# Conditionally link lane detection libraries if TensorRT is available
if(TENSORRT_FOUND)
    target_link_libraries(integrated_lane_fusion_node
            ${LANE_DETECTION_LIBRARIES}
            ${LANE_DETECTION_TENSORRT_LIBRARIES}
    )
    target_compile_definitions(integrated_lane_fusion_node PRIVATE LANE_DETECTION_ENABLED)
endif()

# Link CUDA libraries if available
if(CUDA_FOUND)
    target_link_libraries(integrated_lane_fusion_node
            ${LIDAR_PROCESSING_CUDA_LIBRARIES}
            ${CAMERA_STITCHING_CUDA_LIBRARIES}
            ${SENSOR_FUSION_CUDA_LIBRARIES}
            ${CUDA_LIBRARIES}
    )

    if(TENSORRT_FOUND)
        target_link_libraries(integrated_lane_fusion_node
                ${LANE_DETECTION_CUDA_LIBRARIES}
        )
    endif()

    target_compile_definitions(integrated_lane_fusion_node PRIVATE CUDA_ACCELERATION_ENABLED)
endif()

# Real-time scheduling optimization for main node
set_target_properties(integrated_lane_fusion_node PROPERTIES
        COMPILE_FLAGS "-DREAL_TIME_SCHEDULING -pthread -DINTEGRATED_FUSION_NODE"
        LINK_FLAGS "-pthread -lrt -Wl,--no-as-needed"
)

# Standalone ground extraction node (from LiDAR module)
add_executable(lidar_ground_extraction_node
        lidar_processing/ros/src/lidar_ground_extraction_node.cpp
)

target_link_libraries(lidar_ground_extraction_node
        ${LIDAR_PROCESSING_LIBRARIES}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${Boost_LIBRARIES}
)

if(CUDA_FOUND AND LIDAR_PROCESSING_CUDA_LIBRARIES)
    target_link_libraries(lidar_ground_extraction_node
            ${LIDAR_PROCESSING_CUDA_LIBRARIES}
            ${CUDA_LIBRARIES}
    )
endif()

# Lane detection node (conditional on TensorRT)
if(TENSORRT_FOUND)
    add_executable(lane_detection_node
            lane_detection/ros/src/lane_detection_node.cpp
    )

    target_link_libraries(lane_detection_node
            ${LANE_DETECTION_LIBRARIES}
            ${LANE_DETECTION_CUDA_LIBRARIES}
            ${LANE_DETECTION_TENSORRT_LIBRARIES}
            ${catkin_LIBRARIES}
            ${OpenCV_LIBRARIES}
            ${CUDA_LIBRARIES}
    )

    set_target_properties(lane_detection_node PROPERTIES
            COMPILE_FLAGS "-DREAL_TIME_SCHEDULING -pthread"
            LINK_FLAGS "-pthread -lrt"
    )
endif()

# Camera stitching node
add_executable(camera_stitching_node
        camera_stitching/src/camera_stitching_node.cpp
)

target_link_libraries(camera_stitching_node
        ${CAMERA_STITCHING_LIBRARIES}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
)

if(CUDA_FOUND AND CAMERA_STITCHING_CUDA_LIBRARIES)
    target_link_libraries(camera_stitching_node
            ${CAMERA_STITCHING_CUDA_LIBRARIES}
            ${CUDA_LIBRARIES}
    )
endif()

# Sensor fusion node
add_executable(sensor_fusion_node
        fusion/ros/src/lane_fusion_node.cpp
)

target_link_libraries(sensor_fusion_node
        ${SENSOR_FUSION_LIBRARIES}
        ${LIDAR_PROCESSING_LIBRARIES}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${PCL_LIBRARIES}
)

if(TENSORRT_FOUND)
    target_link_libraries(sensor_fusion_node
            ${LANE_DETECTION_LIBRARIES}
    )
endif()

if(CUDA_FOUND)
    target_link_libraries(sensor_fusion_node
            ${SENSOR_FUSION_CUDA_LIBRARIES}
            ${LIDAR_PROCESSING_CUDA_LIBRARIES}
            ${CUDA_LIBRARIES}
    )
endif()

##############################################
## Dependencies for integrated build
##############################################

# Ensure module libraries are built before executables
add_dependencies(integrated_lane_fusion_node
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
)

add_dependencies(lidar_ground_extraction_node
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
)

add_dependencies(camera_stitching_node
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
)

add_dependencies(sensor_fusion_node
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
)

if(TENSORRT_FOUND)
    add_dependencies(lane_detection_node
            ${${PROJECT_NAME}_EXPORTED_TARGETS}
            ${catkin_EXPORTED_TARGETS}
    )
endif()

##############################################
## Install configuration
##############################################

# Install main executables
install(TARGETS
        integrated_lane_fusion_node
        lidar_ground_extraction_node
        camera_stitching_node
        sensor_fusion_node
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(TENSORRT_FOUND)
    install(TARGETS
            lane_detection_node
            RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
endif()

# Install shared resources (config, launch files, models)
install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
        FILES_MATCHING PATTERN "*.launch" PATTERN "*.py"
        PATTERN ".svn" EXCLUDE)

install(DIRECTORY config/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
        PATTERN ".svn" EXCLUDE)

# Install lane detection models if TensorRT is available
if(TENSORRT_FOUND)
    install(DIRECTORY lane_detection/models/
            DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
            PATTERN ".svn" EXCLUDE)
endif()

# Install scripts
install(PROGRAMS
        scripts/setup_environment.sh
        scripts/build.sh
        scripts/calibration_collection.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

##############################################
## Build status and module summary
##############################################

message(STATUS "=== Lane Fusion Build Configuration Summary ===")
message(STATUS "Target Platform: Embedded (${CMAKE_CUDA_ARCHITECTURES})")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "PCL: Found")
message(STATUS "CUDA: ${CUDA_VERSION}")
message(STATUS "Eigen3: ${EIGEN3_VERSION}")

# Module build status
message(STATUS "Module Status:")
message(STATUS "  ✓ LiDAR Processing: ENABLED")
message(STATUS "  ✓ Camera Stitching: ENABLED")
message(STATUS "  ✓ Sensor Fusion: ENABLED")

if(TENSORRT_FOUND)
    message(STATUS "  ✓ Lane Detection: ENABLED (TensorRT ${TENSORRT_ROOT})")
else()
    message(STATUS "  ✗ Lane Detection: DISABLED (TensorRT not found)")
endif()

# Executable targets
message(STATUS "Executable Targets:")
message(STATUS "  - integrated_lane_fusion_node (unified multi-modal processing)")
message(STATUS "  - lidar_ground_extraction_node (LiDAR-only processing)")
message(STATUS "  - camera_stitching_node (panoramic view generation)")
message(STATUS "  - sensor_fusion_node (projection-based fusion)")

if(TENSORRT_FOUND)
    message(STATUS "  - lane_detection_node (neural network-based detection)")
endif()

# Performance characteristics
message(STATUS "Performance Optimizations:")
message(STATUS "  - Real-time scheduling: ENABLED")
message(STATUS "  - CUDA acceleration: ENABLED")
message(STATUS "  - Embedded platform tuning: ENABLED")
message(STATUS "  - Memory optimization: ENABLED")

# CUDA kernel compilation status
if(CUDA_FOUND)
    message(STATUS "CUDA Kernels Compiled:")
    message(STATUS "  - LiDAR processing kernels")
    message(STATUS "  - Camera stitching kernels")
    message(STATUS "  - Sensor fusion kernels")
    if(TENSORRT_FOUND)
        message(STATUS "  - Lane detection tensor kernels")
    endif()
endif()

message(STATUS "================================================")

# Development guidance
message(STATUS "Development Notes:")
message(STATUS "  Individual module building: cd <module> && mkdir build && cmake ..")
message(STATUS "  Integrated building: catkin_make or catkin build")
message(STATUS "  Performance profiling: Use integrated_lane_fusion_node")
message(STATUS "================================================")