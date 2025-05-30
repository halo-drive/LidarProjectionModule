cmake_minimum_required(VERSION 3.16)

# Set CMake policies
cmake_policy(SET CMP0074 NEW)

# LiDAR processing module - standalone and integrated build support
set(MODULE_NAME "lidar_processing")
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

    # PCL for point cloud processing (no OpenCV needed for LiDAR module)
    find_package(PCL REQUIRED COMPONENTS
            common io features search kdtree segmentation filters)

    find_package(Eigen3 REQUIRED)
    find_package(Boost REQUIRED COMPONENTS system thread)

    # CUDA for embedded platform acceleration
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        enable_language(CUDA)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 72 75 87)
        endif()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-extended-lambda")
        message(STATUS "CUDA ${CUDA_VERSION} enabled for architectures: ${CMAKE_CUDA_ARCHITECTURES}")
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
            ${PCL_INCLUDE_DIRS}
            ${EIGEN3_INCLUDE_DIRS}
            ${Boost_INCLUDE_DIRS}
    )

    if(CUDA_FOUND)
        include_directories(${CUDA_INCLUDE_DIRS})
    endif()
endif()

# Compiler definitions and optimizations
add_definitions(-DLIDAR_PROCESSING_MODULE -DPHASE3_BUILD_VALIDATION -DEMBEDDED_PLATFORM)
add_definitions(${PCL_DEFINITIONS})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-O3 -march=native -std=c++14)
    add_compile_options(-ffast-math -funroll-loops)
endif()

# CUDA kernels for point cloud acceleration
if(CUDA_FOUND OR (DEFINED WORKSPACE_CUDA_FOUND AND WORKSPACE_CUDA_FOUND))
    if(STANDALONE_BUILD)
        add_library(${MODULE_NAME}_cuda_kernels
                cuda/test_kernels.cu
        )
        set(CUDA_TARGET_NAME ${MODULE_NAME}_cuda_kernels)
    else()
        add_library(lane_fusion_${MODULE_NAME}_cuda_kernels
                ${CMAKE_CURRENT_SOURCE_DIR}/cuda/test_kernels.cu
        )
        set(CUDA_TARGET_NAME lane_fusion_${MODULE_NAME}_cuda_kernels)
    endif()

    target_link_libraries(${CUDA_TARGET_NAME}
            ${CUDA_LIBRARIES}
    )

    set_target_properties(${CUDA_TARGET_NAME} PROPERTIES
            CUDA_RUNTIME_LIBRARY Static
            POSITION_INDEPENDENT_CODE ON
    )

    if(DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(${CUDA_TARGET_NAME} PROPERTIES
                CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
    endif()

    set(HAS_CUDA_KERNELS TRUE)
else()
    set(HAS_CUDA_KERNELS FALSE)
    message(STATUS "CUDA not available - building CPU-only version")
endif()

# Point cloud processing library
if(STANDALONE_BUILD)
    set(CORE_LIB_NAME ${MODULE_NAME}_core)
else()
    set(CORE_LIB_NAME lane_fusion_${MODULE_NAME}_core)
endif()

add_library(${CORE_LIB_NAME}
        src/point_cloud_proc.cpp
)

if(STANDALONE_BUILD)
    target_link_libraries(${CORE_LIB_NAME}
            ${ROSCPP_LIBRARY}
            ${ROSTIME_LIBRARY}
            ${PCL_LIBRARIES}
            ${Boost_LIBRARIES}
    )
else()
    target_link_libraries(${CORE_LIB_NAME}
            ${PCL_LIBRARIES}
            ${Boost_LIBRARIES}
    )
endif()

if(HAS_CUDA_KERNELS)
    target_link_libraries(${CORE_LIB_NAME} ${CUDA_TARGET_NAME})
    target_compile_definitions(${CORE_LIB_NAME} PRIVATE CUDA_ACCELERATION)
endif()

# Ground extraction library
if(STANDALONE_BUILD)
    set(GROUND_LIB_NAME ${MODULE_NAME}_ground_extraction)
else()
    set(GROUND_LIB_NAME lane_fusion_${MODULE_NAME}_ground_extraction)
endif()

add_library(${GROUND_LIB_NAME}
        src/ground_extraction.cpp
)

target_link_libraries(${GROUND_LIB_NAME}
        ${CORE_LIB_NAME}
)

if(STANDALONE_BUILD)
    target_link_libraries(${GROUND_LIB_NAME}
            ${PCL_LIBRARIES}
    )
endif()

# Embedded platform optimizations
set_target_properties(${CORE_LIB_NAME} PROPERTIES
        COMPILE_FLAGS "-DREAL_TIME_PROCESSING"
        LINK_FLAGS "-Wl,--gc-sections"
)

set_target_properties(${GROUND_LIB_NAME} PROPERTIES
        COMPILE_FLAGS "-DGROUND_EXTRACTION_MODULE"
        LINK_FLAGS "-Wl,--gc-sections"
)

# Test executable for standalone builds
if(STANDALONE_BUILD)
    add_executable(${MODULE_NAME}_test
            src/lidar_ground_extraction_node.cpp
    )

    target_link_libraries(${MODULE_NAME}_test
            ${GROUND_LIB_NAME}
            ${CORE_LIB_NAME}
            ${ROSCPP_LIBRARY}
            ${ROSTIME_LIBRARY}
            ${PCL_LIBRARIES}
    )
endif()

# Export variables for parent project
if(NOT STANDALONE_BUILD)
    set(LIDAR_PROCESSING_LIBRARIES
            ${CORE_LIB_NAME}
            ${GROUND_LIB_NAME}
            PARENT_SCOPE
    )

    if(HAS_CUDA_KERNELS)
        set(LIDAR_PROCESSING_CUDA_LIBRARIES
                ${CUDA_TARGET_NAME}
                PARENT_SCOPE
        )
    endif()

    set(LIDAR_PROCESSING_INCLUDE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            PARENT_SCOPE
    )
endif()

# Configuration summary
message(STATUS "=== ${MODULE_NAME} Configuration Summary ===")
message(STATUS "Build mode: ${CMAKE_BUILD_TYPE}")
message(STATUS "Standalone: ${STANDALONE_BUILD}")
message(STATUS "PCL: ${PCL_VERSION}")
message(STATUS "CUDA: ${HAS_CUDA_KERNELS}")
if(HAS_CUDA_KERNELS)
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "Libraries: ${CORE_LIB_NAME}, ${GROUND_LIB_NAME}")
if(STANDALONE_BUILD)
    message(STATUS "Test executable: ${MODULE_NAME}_test")
endif()
message(STATUS "Phase 3-4 build validation: CONFIGURED")
message(STATUS "==========================================")