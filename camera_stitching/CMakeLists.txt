cmake_minimum_required(VERSION 3.16)
project(camera_stitching VERSION 1.0.0 LANGUAGES CXX CUDA)

#
# ─── Build options ─────────────────────────────────────────────────────
#
option(BUILD_SHARED_LIBS "Build shared (.so) instead of static" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#
# ─── Dependencies ──────────────────────────────────────────────────────
#
# 1. ROS (choose catkin if we are inside a workspace, else pkg-config)
find_package(catkin QUIET COMPONENTS roscpp sensor_msgs cv_bridge std_msgs)

if (catkin_FOUND)
    message(STATUS "catkin workspace detected")
    set(ROS_LIBRARIES    ${catkin_LIBRARIES})
    set(ROS_INCLUDE_DIRS ${catkin_INCLUDE_DIRS})
else()
    message(STATUS "stand-alone build – resolving ROS libs via pkg-config")
    execute_process(COMMAND pkg-config --cflags --libs roscpp OUTPUT_VARIABLE _rosflags OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX REPLACE "-I([^ ]+)" "\\1" ROS_INCLUDE_DIRS "${_rosflags}")
    string(REGEX MATCHALL "-l[^ ]+"   _roslibs "${_rosflags}")
    foreach(flag ${_roslibs})
        string(SUBSTRING ${flag} 2 -1 lib)
        list(APPEND ROS_LIBRARIES ${lib})
    endforeach()
endif()

# 2. OpenCV, Boost, CUDA (optional)
find_package(OpenCV  REQUIRED COMPONENTS core imgproc calib3d imgcodecs highgui)
find_package(Boost   REQUIRED COMPONENTS system thread)
find_package(CUDA QUIET)
if (CUDA_FOUND)
    enable_language(CUDA)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 72 75 87)
    endif()
endif()

include_directories(
        ${ROS_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

#
# ─── Library  ──────────────────────────────────────────────────────────
#
set(SRC_FILES
        src/camera_synchronizer.cpp
        src/panoramic_stitcher.cpp
)

if (CUDA_FOUND)
    list(APPEND SRC_FILES cuda/image_kernels.cu)
endif()

add_library(camera_stitching ${SRC_FILES})

target_link_libraries(camera_stitching
        ${ROS_LIBRARIES}
        ${OpenCV_LIBRARIES}
        Boost::system Boost::thread
        ${CUDA_LIBRARIES}
)

#
# ─── Node executable  ─────────────────────────────────────────────────
#
add_executable(camera_stitching_node src/camera_stitching_node.cpp)
target_link_libraries(camera_stitching_node camera_stitching)

#
# ─── Installation / export (optional) ─────────────────────────────────
#
install(TARGETS camera_stitching camera_stitching_node
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

#
# ─── Catkin package metadata (ignored in stand-alone mode) ────────────
#
if (catkin_FOUND)
    catkin_package(INCLUDE_DIRS include
            LIBRARIES camera_stitching
            CATKIN_DEPENDS roscpp sensor_msgs cv_bridge std_msgs
            DEPENDS OpenCV Boost)
endif()
