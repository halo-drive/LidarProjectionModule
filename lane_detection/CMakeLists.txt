cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0074 NEW)              # allow CUDA_ROOT / TensorRT root hints

project(lane_detection VERSION 1.0.0 LANGUAGES CXX CUDA)
set(MODULE_NAME ${PROJECT_NAME})

# -----------------------------------------------------------------------------
# Detect build context
#   • STANDALONE_BUILD = ON  → user forced it
#   • building under catkin?  (CATKIN_TOPLEVEL is defined only in catkin workspaces)
# -----------------------------------------------------------------------------
if(NOT DEFINED STANDALONE_BUILD)
    if(DEFINED CATKIN_TOPLEVEL)         # we are *inside* a catkin workspace
        set(STANDALONE_BUILD TRUE)        # treat as stand-alone package
    else()
        # integrated if called from umbrella project
        set(STANDALONE_BUILD FALSE)
    endif()
endif()
if(STANDALONE_BUILD)
    project(${MODULE_NAME})
else()
    message(STATUS "Building ${MODULE_NAME} as part of umbrella project")
endif()



# ───────────────────────────── Options ────────────────────────────────
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ───────────────────────────── ROS deps ───────────────────────────────
find_package(catkin QUIET COMPONENTS
        roscpp cv_bridge sensor_msgs std_msgs)

if (catkin_FOUND)
    message(STATUS "catkin workspace detected")
    set(ROS_INCLUDE_DIRS ${catkin_INCLUDE_DIRS})
    set(ROS_LIBRARIES    ${catkin_LIBRARIES})
else()
    message(STATUS "stand-alone build – resolving ROS libs via pkg-config")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ROSCPP REQUIRED roscpp)
    list(APPEND ROS_INCLUDE_DIRS ${ROSCPP_INCLUDE_DIRS})
    list(APPEND ROS_LIBRARIES    ${ROSCPP_LIBRARIES})
endif()

# ────────────────────────── Other dependencies ───────────────────────
find_package(OpenCV REQUIRED COMPONENTS
        core imgproc features2d calib3d imgcodecs highgui)
find_package(Boost  REQUIRED COMPONENTS system filesystem thread)

# CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 72 75 87)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-extended-lambda --expt-relaxed-constexpr --maxrregcount=32")

# TensorRT (manually locate)
set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT root directory")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} /usr /usr/local
        PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY_INFER    nvinfer
        HINTS ${TENSORRT_ROOT} /usr /usr/local
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNX     nvonnxparser
        HINTS ${TENSORRT_ROOT} /usr /usr/local
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_PLUGIN   nvinfer_plugin
        HINTS ${TENSORRT_ROOT} /usr /usr/local
        PATH_SUFFIXES lib lib64 lib/x64)

if (TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY_INFER AND TENSORRT_LIBRARY_ONNX)
    set(TENSORRT_FOUND TRUE)
else()
    message(FATAL_ERROR "TensorRT not found – set TENSORRT_ROOT or install it.")
endif()

# ───────────────────────── Include directories ───────────────────────
include_directories(
        ${ROS_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ───────────────────────── CUDA kernels lib ───────────────────────────
cuda_add_library(lane_detection_cuda_kernels
        cuda/tensor_kernels.cu)
target_link_libraries(lane_detection_cuda_kernels ${CUDA_LIBRARIES})

# ───────────────────────── Core utility libs ──────────────────────────
add_library(lane_detection_tensor_utils  src/tensor_utils.cpp)
target_link_libraries(lane_detection_tensor_utils
        lane_detection_cuda_kernels
        ${CUDA_LIBRARIES}
        ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNX} ${TENSORRT_LIBRARY_PLUGIN}
        Boost::system Boost::filesystem Boost::thread
        ${OpenCV_LIBRARIES}
        ${ROS_LIBRARIES})

add_library(lane_detection_yolo_detector src/yolo_detector.cpp)
target_link_libraries(lane_detection_yolo_detector
        lane_detection_tensor_utils
        ${CUDA_LIBRARIES}
        ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNX}
        ${OpenCV_LIBRARIES}
        ${ROS_LIBRARIES})

add_library(lane_detection_segmentation  src/lane_segmentation.cpp)
target_link_libraries(lane_detection_segmentation
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${ROS_LIBRARIES})

# ───────────────────────── Node executable ────────────────────────────
add_executable(lane_detection_node
        src/lane_detection_node.cpp)

target_link_libraries(lane_detection_node
        lane_detection_yolo_detector
        lane_detection_segmentation
        ${ROS_LIBRARIES}
        ${OpenCV_LIBRARIES})

# ───────────────────────── Installation / export ──────────────────────
install(TARGETS
        lane_detection_cuda_kernels
        lane_detection_tensor_utils
        lane_detection_yolo_detector
        lane_detection_segmentation
        lane_detection_node
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME  DESTINATION lib/${PROJECT_NAME})

if (catkin_FOUND)
    catkin_package(
            INCLUDE_DIRS include
            LIBRARIES lane_detection_tensor_utils
            lane_detection_yolo_detector
            lane_detection_segmentation
            CATKIN_DEPENDS roscpp cv_bridge sensor_msgs std_msgs
            DEPENDS OpenCV Boost
    )
endif()
