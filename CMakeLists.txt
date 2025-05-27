cmake_minimum_required(VERSION 3.0.2)
project(lane_fusion)

## Compile as C++14, required for TensorRT
add_compile_options(-std=c++14)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        sensor_msgs
        geometry_msgs
        visualization_msgs
        image_transport
        cv_bridge
        tf2
        tf2_ros
)

## System dependencies
find_package(OpenCV 4 REQUIRED)
find_package(PCL 1.8 REQUIRED)

# CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# TensorRT - Multiple search paths for different installations
set(TENSORRT_ROOT "/usr/src/tensorrt" CACHE PATH "TensorRT root directory")

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
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY_INFER OR NOT TENSORRT_LIBRARY_ONNXPARSER)
    message(FATAL_ERROR "TensorRT not found. Please set TENSORRT_ROOT or install TensorRT")
endif()

message(STATUS "Found TensorRT:")
message(STATUS "  Include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "  Libraries: ${TENSORRT_LIBRARY_INFER}")

## CUDA architecture settings
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)  # RTX 20xx series - adjust as needed
endif()

## Define CUDA compilation flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES})

###################################
## catkin specific configuration ##
###################################
catkin_package(
        INCLUDE_DIRS
        lane_detection/include        # Add lane_detection headers
        ${TENSORRT_INCLUDE_DIR}
        LIBRARIES
        ${PROJECT_NAME}_tensor_utils
        ${PROJECT_NAME}_yolo_detector
        ${PROJECT_NAME}_lane_segmentation
        # Future libraries (commented until implemented):
        # ${PROJECT_NAME}_lidar_processing
        # ${PROJECT_NAME}_fusion
        # ${PROJECT_NAME}_utils
        # ${PROJECT_NAME}_calibration
        CATKIN_DEPENDS
        roscpp
        rospy
        std_msgs
        sensor_msgs
        geometry_msgs
        visualization_msgs
        image_transport
        cv_bridge
        tf2
        tf2_ros
        DEPENDS
        OpenCV
        PCL
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
        lane_detection/include     # Lane detection module headers (ACTIVE)
        # Future module includes (commented until implemented):
        # lidar_processing/include
        # fusion/include
        # calibration/include
        # utils/include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
)

###################
## CUDA Libraries ##
###################

# CUDA Tensor Utilities (ACTIVE)
cuda_add_library(${PROJECT_NAME}_tensor_utils
        lane_detection/src/tensor_utils.cpp
        lane_detection/cuda/tensor_kernels.cu
)

target_link_libraries(${PROJECT_NAME}_tensor_utils
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${CUDA_LIBRARIES}          # Fixed typo: removed 'z'
        ${TENSORRT_LIBRARY_INFER}
        ${TENSORRT_LIBRARY_ONNXPARSER}
        ${TENSORRT_LIBRARY_PLUGIN}
        cublas
        curand
        cudnn
)

###################
## Lane Detection Libraries (ACTIVE) ##
###################

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

# Remove empty combined library - use components directly

###################
## Future Libraries (Commented until implemented) ##
###################

# Uncomment when lidar_processing module is ready:
# add_library(${PROJECT_NAME}_lidar_processing
#         lidar_processing/src/point_cloud_proc.cpp
#         lidar_processing/src/ground_extraction.cpp
# )
#
# target_link_libraries(${PROJECT_NAME}_lidar_processing
#         ${catkin_LIBRARIES}
#         ${PCL_LIBRARIES}
# )

# Uncomment when fusion module is ready:
# add_library(${PROJECT_NAME}_fusion
#         fusion/src/projection.cpp
#         fusion/src/lane_model.cpp
# )
#
# target_link_libraries(${PROJECT_NAME}_fusion
#         ${catkin_LIBRARIES}
#         ${OpenCV_LIBRARIES}
#         ${PCL_LIBRARIES}
# )

# Uncomment when calibration module is ready:
# add_library(${PROJECT_NAME}_calibration
#         calibration/src/camera_calibrator.cpp
#         calibration/src/lidar_calibrator.cpp
#         calibration/src/extrinsic_calibrator.cpp
# )
#
# target_link_libraries(${PROJECT_NAME}_calibration
#         ${catkin_LIBRARIES}
#         ${OpenCV_LIBRARIES}
#         ${PCL_LIBRARIES}
# )

# Uncomment when utils module is ready:
# add_library(${PROJECT_NAME}_utils
#         utils/src/time_sync.cpp
#         utils/src/memory_management.cpp
#         utils/src/visualization.cpp
# )
#
# target_link_libraries(${PROJECT_NAME}_utils
#         ${catkin_LIBRARIES}
#         ${OpenCV_LIBRARIES}
# )

###################
## Executables ##
###################

# Lane Detection Node with full TensorRT support (ACTIVE)
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

# Future executables (commented until modules are ready):
# add_executable(calibration_node
#         calibration/tools/calibration_node.cpp
# )
#
# target_link_libraries(calibration_node
#         ${PROJECT_NAME}_calibration
#         ${catkin_LIBRARIES}
# )

# add_executable(lane_fusion_node
#         fusion/src/lane_fusion_node.cpp
# )
#
# target_link_libraries(lane_fusion_node
#         ${PROJECT_NAME}_fusion
#         ${PROJECT_NAME}_lane_detection
#         ${PROJECT_NAME}_lidar_processing
#         ${catkin_LIBRARIES}
# )

###################
## Install ##
###################

# Mark executables for installation
install(TARGETS
        lane_detection_node
        # Future executables:
        # calibration_node
        # lane_fusion_node
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark libraries for installation
install(TARGETS
        ${PROJECT_NAME}_yolo_detector
        ${PROJECT_NAME}_lane_segmentation
        ${PROJECT_NAME}_tensor_utils
        # Future libraries:
        # ${PROJECT_NAME}_lidar_processing
        # ${PROJECT_NAME}_fusion
        # ${PROJECT_NAME}_calibration
        # ${PROJECT_NAME}_utils
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

# Install headers
install(DIRECTORY lane_detection/include/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}/lane_detection
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)