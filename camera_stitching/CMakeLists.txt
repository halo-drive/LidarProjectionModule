cmake_minimum_required(VERSION 3.16)
project(camera_stitching VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

###############################################################################
# Core Dependencies - Always Required
###############################################################################
find_package(catkin REQUIRED COMPONENTS
        roscpp
        sensor_msgs
        cv_bridge
        std_msgs
        image_transport
        message_filters
        visualization_msgs
)

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d imgcodecs highgui features2d)
find_package(Boost REQUIRED COMPONENTS system thread)

###############################################################################
# Optional Dependencies - Graceful Degradation
###############################################################################

# CUDA Detection
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 72 75 87)
    endif()
    add_definitions(-DCUDA_ENABLED)
    message(STATUS "CUDA support: ENABLED")
else()
    message(STATUS "CUDA support: DISABLED (optional)")
endif()

# TensorRT Detection
if(CUDA_FOUND)
    set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT root directory")
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS ${TENSORRT_ROOT} /workspace/TensorRT-8.5.2.2
            PATH_SUFFIXES include)
    find_library(TENSORRT_LIBRARY_INFER nvinfer
            HINTS ${TENSORRT_ROOT} /workspace/TensorRT-8.5.2.2
            PATH_SUFFIXES lib lib64)

    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY_INFER)
        set(TENSORRT_FOUND TRUE)
        add_definitions(-DTENSORRT_ENABLED)
        message(STATUS "TensorRT support: ENABLED")
    else()
        set(TENSORRT_FOUND FALSE)
        message(STATUS "TensorRT support: DISABLED (optional)")
    endif()
else()
    set(TENSORRT_FOUND FALSE)
    message(STATUS "TensorRT support: DISABLED (requires CUDA)")
endif()

###############################################################################
# Include Directories
###############################################################################
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

if(TENSORRT_FOUND)
    include_directories(${TENSORRT_INCLUDE_DIR})
endif()

###############################################################################
# Core Sources - Always Compiled
###############################################################################
set(CAMERA_STITCHING_SOURCES
        src/camera_synchronizer.cpp
        src/panoramic_stitcher.cpp
)

# Add CUDA sources if available
if(CUDA_FOUND)
    list(APPEND CAMERA_STITCHING_SOURCES cuda/image_kernels.cu)
endif()

###############################################################################
# Dependency Detection and Optional Integration
###############################################################################

# Function to safely check and add source files
function(add_optional_sources target_list source_description)
    foreach(source_file ${ARGN})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
            list(APPEND ${target_list} ${source_file})
            message(STATUS "Added ${source_description}: ${source_file}")
        else()
            message(STATUS "Optional ${source_description} not found: ${source_file}")
        endif()
    endforeach()
    set(${target_list} ${${target_list}} PARENT_SCOPE)
endfunction()

# Attempt to find sibling modules in catkin workspace
set(OPTIONAL_SOURCES "")

# Look for lane_detection sources in catkin workspace context
add_optional_sources(OPTIONAL_SOURCES "lane_detection"
        ../lane_detection/src/tensor_utils.cpp
        ../lane_detection/src/yolo_detector.cpp
        ../lane_detection/src/lane_segmentation.cpp
)

# Look for calibration sources
add_optional_sources(OPTIONAL_SOURCES "calibration"
        ../calibration/src/camera_calibrator.cpp
        ../calibration/src/lidar_calibrator.cpp
        ../calibration/src/extrinsic_calibrator.cpp
)

# Look for utils sources
add_optional_sources(OPTIONAL_SOURCES "utils"
        ../utils/src/memory_management.cpp
        ../utils/src/time_sync.cpp
        ../utils/src/visualization.cpp
)

# Add found optional sources to main source list
list(APPEND CAMERA_STITCHING_SOURCES ${OPTIONAL_SOURCES})

###############################################################################
# Stub Implementations for Missing Dependencies
###############################################################################

# Create stub source file for missing dependencies
set(STUB_IMPLEMENTATIONS "${CMAKE_CURRENT_BINARY_DIR}/dependency_stubs.cpp")
file(WRITE ${STUB_IMPLEMENTATIONS} "
// Auto-generated stub implementations for missing dependencies
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <memory>

// Stub implementations to satisfy linker when dependencies are missing
#ifndef TENSORRT_ENABLED
namespace lane_detection {
    class PerformanceMonitor {
    public:
        PerformanceMonitor() {}
        void startTiming(const std::string& operation) {}
        void endTiming(const std::string& operation) {}
        void recordInferenceTime(double time_ms) {}
        double getCurrentFPS() const { return 0.0; }
        double getAverageInferenceTime() const { return 0.0; }
        double getMaxInferenceTime() const { return 0.0; }
        double getMinInferenceTime() const { return 0.0; }
        size_t getCurrentMemoryUsage() const { return 0; }
        void printStatistics() const {}
        void reset() {}
    };

    namespace utils {
        void printCudaDeviceInfo() {
            ROS_WARN(\"CUDA support not compiled - device info unavailable\");
        }
        bool checkCudaCapability(int major, int minor) {
            ROS_WARN(\"CUDA support not compiled - capability check unavailable\");
            return false;
        }
    }
}
#endif

// Placeholder namespace for calibration if not available
namespace calibration {
    // Stubs will be added here if calibration module is missing
}

// Placeholder namespace for memory utils if not available
namespace lane_fusion { namespace utils {
    // Stubs will be added here if utils module is missing
}}
")

# Always include stub implementations
list(APPEND CAMERA_STITCHING_SOURCES ${STUB_IMPLEMENTATIONS})

###############################################################################
# Library Creation
###############################################################################
add_library(camera_stitching ${CAMERA_STITCHING_SOURCES})

target_link_libraries(camera_stitching
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        Boost::system
        Boost::thread
)

if(CUDA_FOUND)
    target_link_libraries(camera_stitching ${CUDA_LIBRARIES})
endif()

if(TENSORRT_FOUND)
    target_link_libraries(camera_stitching ${TENSORRT_LIBRARY_INFER})
endif()

###############################################################################
# Executable Creation
###############################################################################
add_executable(camera_stitching_node src/camera_stitching_node.cpp)

target_link_libraries(camera_stitching_node
        camera_stitching
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

###############################################################################
# Catkin Package Declaration
###############################################################################
catkin_package(
        INCLUDE_DIRS include
        LIBRARIES camera_stitching
        CATKIN_DEPENDS
        roscpp
        sensor_msgs
        cv_bridge
        std_msgs
        image_transport
        message_filters
        visualization_msgs
        DEPENDS OpenCV Boost
)

###############################################################################
# Installation - Fixed destination paths
###############################################################################
install(TARGETS camera_stitching camera_stitching_node
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

###############################################################################
# Build Status Report
###############################################################################
message(STATUS "=== Camera Stitching Build Configuration ===")
message(STATUS "CUDA Found: ${CUDA_FOUND}")
message(STATUS "TensorRT Found: ${TENSORRT_FOUND}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "Optional sources found: ${OPTIONAL_SOURCES}")
message(STATUS "Total sources: ${CAMERA_STITCHING_SOURCES}")
message(STATUS "==========================================")