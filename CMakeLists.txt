###############################################################################
# LidarProjectionLane/CMakeLists.txt
# Umbrella catkin package: 'lane_fusion'
###############################################################################

cmake_minimum_required(VERSION 3.0.2)
project(lane_fusion)

###############################################################################
# 0. Global compile flags
###############################################################################
add_compile_options(-std=c++14)

###############################################################################
# 1. catkin + common dependencies
###############################################################################
find_package(catkin REQUIRED)
# (no catkin_package() yet – we’ll call it after adding sub-dirs)

find_package(OpenCV REQUIRED COMPONENTS           # <-- NO xfeatures2d
        core imgproc calib3d stitching imgcodecs highgui)

find_package(PCL REQUIRED COMPONENTS
        common io features search kdtree segmentation filters)

find_package(Eigen3  REQUIRED)
find_package(Boost   REQUIRED COMPONENTS system filesystem thread)
find_package(CUDA    REQUIRED)
enable_language(CUDA)

# propagate CUDA availability to children (they check this variable)
set(WORKSPACE_CUDA_FOUND TRUE CACHE BOOL "CUDA available in umbrella" FORCE)

# --- TensorRT --------------------------------------------------------------
set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT root dir")

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT}
        PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY_INFER  nvinfer
        HINTS ${TENSORRT_ROOT}
        PATH_SUFFIXES lib lib64)

find_library(TENSORRT_LIBRARY_ONNX   nvonnxparser
        HINTS ${TENSORRT_ROOT}
        PATH_SUFFIXES lib lib64)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY_INFER AND TENSORRT_LIBRARY_ONNX)
    set(TENSORRT_FOUND TRUE)
    set(WORKSPACE_TENSORRT_FOUND TRUE  CACHE BOOL  "TensorRT available" FORCE)
    set(WORKSPACE_TENSORRT_ROOT  "${TENSORRT_ROOT}" CACHE PATH "TensorRT root"  FORCE)
else()
    set(TENSORRT_FOUND FALSE)
    set(WORKSPACE_TENSORRT_FOUND FALSE CACHE BOOL  "TensorRT available" FORCE)
endif()
# ---------------------------------------------------------------------------

###############################################################################
# 2.  Add sub-modules (their own CMakeLists handle everything inside)
###############################################################################
add_subdirectory(camera_stitching)
add_subdirectory(lidar_processing)

if (TENSORRT_FOUND)
    add_subdirectory(lane_detection)
else()
    message(WARNING "TensorRT not found – skipping lane_detection build")
endif()

add_subdirectory(fusion)         # sensor_fusion sources live here

###############################################################################
# 3.  lane_fusion meta-package declaration
###############################################################################
catkin_package()                 # umbrella exports nothing but resources

###############################################################################
# 4.  Install shared resources so `roslaunch lane_fusion …` works
###############################################################################
install(DIRECTORY launch config
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        USE_SOURCE_PERMISSIONS
        PATTERN ".svn" EXCLUDE)

install(DIRECTORY scripts
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        USE_SOURCE_PERMISSIONS
        FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
        PATTERN ".svn" EXCLUDE)

###############################################################################
# 5.  Summary
###############################################################################
message(STATUS "============== lane_fusion build ==============")
message(STATUS "OpenCV       : ${OpenCV_VERSION}")
message(STATUS "PCL          : ${PCL_VERSION}")
message(STATUS "CUDA         : ${CUDA_VERSION}")
message(STATUS "TensorRT     : ${TENSORRT_FOUND}")
message(STATUS "------------------------------------------------")
