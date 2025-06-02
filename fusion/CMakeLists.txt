cmake_minimum_required(VERSION 3.16)
project(sensor_fusion VERSION 1.0.0 LANGUAGES CXX CUDA)
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

#=================================================================================
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ───────────── ROS (catkin or pkg-config) ─────────────
find_package(catkin QUIET COMPONENTS roscpp sensor_msgs pcl_ros pcl_conversions)
if(catkin_FOUND)
    set(ROS_INCLUDE_DIRS ${catkin_INCLUDE_DIRS})
    set(ROS_LIBRARIES    ${catkin_LIBRARIES})
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ROSCPP REQUIRED roscpp)
    set(ROS_INCLUDE_DIRS ${ROSCPP_INCLUDE_DIRS})
    set(ROS_LIBRARIES    ${ROSCPP_LIBRARIES})
endif()

# ───────────── Third-party libs ─────────────
find_package(OpenCV REQUIRED)                # 4.2.0 on your box
find_package(PCL     REQUIRED COMPONENTS common io)
find_package(Eigen3  REQUIRED)
find_package(Boost   REQUIRED COMPONENTS system thread)

# ───────────── CUDA (optional) ─────────────
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 72 75 87)
    endif()
endif()

include_directories(
        ${ROS_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ───────────── CUDA kernels lib ─────────────
if(CUDA_FOUND)
    cuda_add_library(sensor_fusion_cuda_kernels
            cuda/projection_kernels.cu
            cuda/lane_fitting_kernels.cu)
    target_link_libraries(sensor_fusion_cuda_kernels
            ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif()

# ───────────── Core C++ library ─────────────
add_library(sensor_fusion_core
        src/projection.cpp
        src/lane_model.cpp)

target_link_libraries(sensor_fusion_core
        ${ROS_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${PCL_LIBRARIES}
        Boost::system Boost::thread
        Eigen3::Eigen
        sensor_fusion_cuda_kernels)

# ───────────── Node executable ─────────────
add_executable(sensor_fusion_node src/lane_fusion_node.cpp)
target_link_libraries(sensor_fusion_node
        sensor_fusion_core
        ${ROS_LIBRARIES}
        ${OpenCV_LIBRARIES})

# ───────────── Install (optional) ─────────────
install(TARGETS
        sensor_fusion_core
        sensor_fusion_node
        sensor_fusion_cuda_kernels
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME  DESTINATION lib/${PROJECT_NAME})

# ───────────── catkin export (ignored stand-alone) ─────────────
if(catkin_FOUND)
    catkin_package(
            INCLUDE_DIRS include
            LIBRARIES sensor_fusion_core
            CATKIN_DEPENDS roscpp sensor_msgs pcl_ros pcl_conversions
            DEPENDS OpenCV PCL Eigen3 Boost)
endif()
