cmake_minimum_required(VERSION 3.16)

# Set CMake policies for CUDA_ROOT compatibility
cmake_policy(SET CMP0074 NEW)

# Lane detection module with TensorRT optimization
set(MODULE_NAME "lane_detection")
message(STATUS "Configuring ${MODULE_NAME} module for Phase 3-4 build validation")

# Determine build context
if(NOT DEFINED PROJECT_NAME OR PROJECT_NAME STREQUAL "")
    project(${MODULE_NAME}_standalone LANGUAGES CXX)
    set(STANDALONE_BUILD TRUE)
    message(STATUS "Standalone build mode - bypassing catkin packaging")

    # Direct ROS library discovery
    find_path(ROS_INCLUDE_DIR ros/ros.h
            HINTS /opt/ros/noetic/include
            REQUIRED)

    find_library(ROSCPP_LIBRARY roscpp
            HINTS /opt/ros/noetic/lib
            REQUIRED)

    find_library(ROSTIME_LIBRARY rostime
            HINTS /opt/ros/noetic/lib
            REQUIRED)

    find_library(CV_BRIDGE_LIBRARY cv_bridge
            HINTS /opt/ros/noetic/lib
            REQUIRED)

    # OpenCV with production-safe core modules
    find_package(OpenCV REQUIRED COMPONENTS
            core
            imgproc
            features2d
            calib3d
            imgcodecs
            highgui
    )

    find_package(Boost REQUIRED COMPONENTS system filesystem thread)

    # CUDA for TensorRT inference acceleration
    find_package(CUDA REQUIRED)
    enable_language(CUDA)

    # Embedded platform CUDA architectures
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 72 75 87)
    endif()

    # TensorRT detection and configuration
    set(TENSORRT_ROOT "/workspace/TensorRT-8.5.2.2" CACHE PATH "TensorRT root directory")

    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES include
            REQUIRED)

    find_library(TENSORRT_LIBRARY_INFER nvinfer
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/x64
            REQUIRED)

    find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/x64
            REQUIRED)

    find_library(TENSORRT_LIBRARY_PLUGIN nvinfer_plugin
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/x64)

    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY_INFER AND TENSORRT_LIBRARY_ONNXPARSER)
        set(TENSORRT_FOUND TRUE)
        message(STATUS "TensorRT found: ${TENSORRT_ROOT}")
    else()
        message(FATAL_ERROR "TensorRT required for lane detection module")
    endif()

else()
    set(STANDALONE_BUILD FALSE)
    message(STATUS "Integrated build mode - part of ${PROJECT_NAME}")
endif()

# Include directories
include_directories(
        include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(STANDALONE_BUILD)
    include_directories(
            ${ROS_INCLUDE_DIR}
            ${OpenCV_INCLUDE_DIRS}
            ${Boost_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
            ${TENSORRT_INCLUDE_DIR}
    )
endif()

# Compiler definitions and embedded platform optimizations
add_definitions(-DLANE_DETECTION_MODULE -DTENSORRT_ENABLED -DPHASE3_BUILD_VALIDATION)
add_definitions(-DEMBEDDED_PLATFORM -DREAL_TIME_INFERENCE)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-O3 -march=native -std=c++14)
    add_compile_options(-ffast-math -funroll-loops -flto)
endif()

# CUDA flags for embedded inference optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --maxrregcount=32")

# CUDA tensor processing kernels
if(STANDALONE_BUILD)
    add_library(${MODULE_NAME}_cuda_kernels
            cuda/tensor_kernels.cu
    )
    set(CUDA_TARGET_NAME ${MODULE_NAME}_cuda_kernels)
else()
    add_library(lane_fusion_${MODULE_NAME}_cuda_kernels
            ${CMAKE_CURRENT_SOURCE_DIR}/cuda/tensor_kernels.cu
    )
    set(CUDA_TARGET_NAME lane_fusion_${MODULE_NAME}_cuda_kernels)
endif()

target_link_libraries(${CUDA_TARGET_NAME}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_CURAND_LIBRARIES}
)

# CUDA properties for embedded deployment
set_target_properties(${CUDA_TARGET_NAME} PROPERTIES
        CUDA_RUNTIME_LIBRARY Static
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
)

if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(${CUDA_TARGET_NAME} PROPERTIES
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
endif()

# TensorRT utilities and engine management
if(STANDALONE_BUILD)
    set(TENSOR_UTILS_NAME ${MODULE_NAME}_tensor_utils)
else()
    set(TENSOR_UTILS_NAME lane_fusion_${MODULE_NAME}_tensor_utils)
endif()

add_library(${TENSOR_UTILS_NAME}
        src/tensor_utils.cpp
)

target_link_libraries(${TENSOR_UTILS_NAME}
        ${CUDA_TARGET_NAME}
)

if(STANDALONE_BUILD)
    target_link_libraries(${TENSOR_UTILS_NAME}
            ${ROSCPP_LIBRARY}
            ${ROSTIME_LIBRARY}
            ${OpenCV_LIBRARIES}
            ${CUDA_LIBRARIES}
            ${TENSORRT_LIBRARY_INFER}
            ${TENSORRT_LIBRARY_ONNXPARSER}
            ${TENSORRT_LIBRARY_PLUGIN}
            cublas
            curand
            cudnn
    )
else()
    target_link_libraries(${TENSOR_UTILS_NAME}
            ${OpenCV_LIBRARIES}
            ${CUDA_LIBRARIES}
            ${TENSORRT_LIBRARY_INFER}
            ${TENSORRT_LIBRARY_ONNXPARSER}
            ${TENSORRT_LIBRARY_PLUGIN}
            cublas
            curand
            cudnn
    )
endif()

# YOLOv8 detector with embedded optimizations
if(STANDALONE_BUILD)
    set(YOLO_DETECTOR_NAME ${MODULE_NAME}_yolo_detector)
else()
    set(YOLO_DETECTOR_NAME lane_fusion_${MODULE_NAME}_yolo_detector)
endif()

add_library(${YOLO_DETECTOR_NAME}
        src/yolo_detector.cpp
)

target_link_libraries(${YOLO_DETECTOR_NAME}
        ${TENSOR_UTILS_NAME}
        ${CUDA_TARGET_NAME}
)

if(STANDALONE_BUILD)
    target_link_libraries(${YOLO_DETECTOR_NAME}
            ${OpenCV_LIBRARIES}
            ${CUDA_LIBRARIES}
            ${TENSORRT_LIBRARY_INFER}
            ${TENSORRT_LIBRARY_ONNXPARSER}
    )
else()
    target_link_libraries(${YOLO_DETECTOR_NAME}
            ${OpenCV_LIBRARIES}
            ${CUDA_LIBRARIES}
            ${TENSORRT_LIBRARY_INFER}
            ${TENSORRT_LIBRARY_ONNXPARSER}
    )
endif()

# Lane segmentation processor
if(STANDALONE_BUILD)
    set(LANE_SEGMENTATION_NAME ${MODULE_NAME}_lane_segmentation)
else()
    set(LANE_SEGMENTATION_NAME lane_fusion_${MODULE_NAME}_lane_segmentation)
endif()

add_library(${LANE_SEGMENTATION_NAME}
        src/lane_segmentation.cpp
)

if(STANDALONE_BUILD)
    target_link_libraries(${LANE_SEGMENTATION_NAME}
            ${ROSCPP_LIBRARY}
            ${ROSTIME_LIBRARY}
            ${OpenCV_LIBRARIES}
            ${Boost_LIBRARIES}
    )
else()
    target_link_libraries(${LANE_SEGMENTATION_NAME}
            ${OpenCV_LIBRARIES}
            ${Boost_LIBRARIES}
    )
endif()

# Embedded platform memory and latency optimizations
set_target_properties(${TENSOR_UTILS_NAME} PROPERTIES
        COMPILE_FLAGS "-DREAL_TIME_CONSTRAINTS -DEMBEDDED_MEMORY_MANAGEMENT"
        LINK_FLAGS "-Wl,--gc-sections -flto"
)

set_target_properties(${YOLO_DETECTOR_NAME} PROPERTIES
        COMPILE_FLAGS "-DREAL_TIME_INFERENCE -DEMBEDDED_INFERENCE_ENGINE"
)

# Test executable for standalone builds
if(STANDALONE_BUILD)
    add_executable(${MODULE_NAME}_test
            src/lane_detection_node.cpp
    )

    target_link_libraries(${MODULE_NAME}_test
            ${YOLO_DETECTOR_NAME}
            ${LANE_SEGMENTATION_NAME}
            ${TENSOR_UTILS_NAME}
            ${ROSCPP_LIBRARY}
            ${ROSTIME_LIBRARY}
            ${CV_BRIDGE_LIBRARY}
            ${OpenCV_LIBRARIES}
    )

    set_target_properties(${MODULE_NAME}_test PROPERTIES
            COMPILE_FLAGS "-DREAL_TIME_SCHEDULING -pthread"
            LINK_FLAGS "-pthread -lrt"
    )
endif()

# Export variables for parent project
if(NOT STANDALONE_BUILD)
    set(LANE_DETECTION_LIBRARIES
            ${TENSOR_UTILS_NAME}
            ${YOLO_DETECTOR_NAME}
            ${LANE_SEGMENTATION_NAME}
            PARENT_SCOPE
    )

    set(LANE_DETECTION_CUDA_LIBRARIES
            ${CUDA_TARGET_NAME}
            PARENT_SCOPE
    )

    set(LANE_DETECTION_INCLUDE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            PARENT_SCOPE
    )

    set(LANE_DETECTION_TENSORRT_LIBRARIES
            ${TENSORRT_LIBRARY_INFER}
            ${TENSORRT_LIBRARY_ONNXPARSER}
            ${TENSORRT_LIBRARY_PLUGIN}
            PARENT_SCOPE
    )
endif()

# Configuration summary
message(STATUS "=== ${MODULE_NAME} Configuration Summary ===")
message(STATUS "Build mode: ${CMAKE_BUILD_TYPE}")
message(STATUS "Standalone: ${STANDALONE_BUILD}")
message(STATUS "TensorRT: ${TENSORRT_FOUND}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Libraries: ${TENSOR_UTILS_NAME}, ${YOLO_DETECTOR_NAME}, ${LANE_SEGMENTATION_NAME}")
message(STATUS "CUDA kernels: ${CUDA_TARGET_NAME}")
if(STANDALONE_BUILD)
    message(STATUS "Test executable: ${MODULE_NAME}_test")
endif()
message(STATUS "TensorRT optimization: ENABLED")
message(STATUS "Embedded constraints: ENABLED")
message(STATUS "Phase 3-4 build validation: CONFIGURED")
message(STATUS "=============================================")