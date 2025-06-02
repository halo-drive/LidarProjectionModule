cmake_minimum_required(VERSION 3.16)
project(lidar_processing VERSION 1.0.0 LANGUAGES CXX CUDA)
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

# ────────────────────────────── Options ───────────────────────────────
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ────────────────────────────── ROS deps ──────────────────────────────
# Prefer catkin if we’re inside a workspace, otherwise fall back to pkg-config.
find_package(catkin QUIET COMPONENTS
        roscpp pcl_ros pcl_conversions sensor_msgs)

if (catkin_FOUND)
    message(STATUS "catkin workspace detected")
    set(ROS_INCLUDE_DIRS ${catkin_INCLUDE_DIRS})
    set(ROS_LIBRARIES    ${catkin_LIBRARIES})
else()
    message(STATUS "stand-alone build – resolving ROS libs via pkg-config")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ROSCPP        REQUIRED roscpp)
    pkg_check_modules(PCL_CONV      REQUIRED pcl_conversions)
    list(APPEND ROS_INCLUDE_DIRS
            ${ROSCPP_INCLUDE_DIRS} ${PCL_CONV_INCLUDE_DIRS})
    list(APPEND ROS_LIBRARIES
            ${ROSCPP_LIBRARIES} ${PCL_CONV_LIBRARIES})
endif()

# ────────────────────────────── Other deps ────────────────────────────
find_package(PCL REQUIRED COMPONENTS
        common io features search kdtree segmentation filters)
find_package(Eigen3 REQUIRED)
find_package(Boost  REQUIRED COMPONENTS system thread)

# CUDA optional
find_package(CUDA QUIET)
if (CUDA_FOUND)
    enable_language(CUDA)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 72 75 87)
    endif()
endif()

include_directories(
        ${ROS_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_definitions(${PCL_DEFINITIONS})

# ─────────────────────────── Core library ─────────────────────────────
set(SRC_CORE
        src/point_cloud_proc.cpp
        src/ground_extraction.cpp
)

if (CUDA_FOUND)
    list(APPEND SRC_CORE
            cuda/ground_extraction.cu
            cuda/voxel_grid.cu)
endif()

add_library(lidar_processing ${SRC_CORE})

target_link_libraries(lidar_processing
        ${ROS_LIBRARIES}
        ${PCL_LIBRARIES}
        Boost::system Boost::thread
        ${CUDA_LIBRARIES}
)

target_compile_definitions(lidar_processing PRIVATE REAL_TIME_PROCESSING)

# ─────────────────────────── Node executable ──────────────────────────
add_executable(lidar_processing_node
        src/lidar_ground_extraction_node.cpp)

target_link_libraries(lidar_processing_node
        lidar_processing
        ${ROS_LIBRARIES}
        ${PCL_LIBRARIES}
)

# ─────────────────────────── Installation  ────────────────────────────
install(TARGETS lidar_processing lidar_processing_node
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME  DESTINATION lib/${PROJECT_NAME})

# ─────────────────────────── catkin export ────────────────────────────
if (catkin_FOUND)
    catkin_package(
            INCLUDE_DIRS include
            LIBRARIES    lidar_processing
            CATKIN_DEPENDS roscpp pcl_ros pcl_conversions sensor_msgs
            DEPENDS PCL Boost
    )
endif()
