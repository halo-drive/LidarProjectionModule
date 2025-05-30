cmake_minimum_required(VERSION 3.16)

# Set CMake policies for CUDA_ROOT compatibility
cmake_policy(SET CMP0074 NEW)

# Camera stitching module - standalone and integrated build support
set(MODULE_NAME "camera_stitching")
message(STATUS "Configuring ${MODULE_NAME} module for Phase 3-4 build validation")

# Determine build context: standalone vs integrated
if(NOT DEFINED PROJECT_NAME OR PROJECT_NAME STREQUAL "")
    project(${MODULE_NAME}_standalone LANGUAGES CXX)
    set(STANDALONE_BUILD TRUE)
    message(STATUS "Standalone build mode - bypassing catkin packaging")

    # Direct ROS library discovery for standalone build
    find_path(ROS_INCLUDE_DIR ros/ros.h
            HINTS /opt/ros/noetic/include
            REQUIRED)

    find_library(ROSCPP_LIBRARY roscpp
            HINTS /opt/ros/noetic/lib
            REQUIRED)

    find_library(ROSTIME_LIBRARY rostime
            HINTS /opt/ros/noetic/lib
            REQUIRED)

    # OpenCV with minimal dependencies for embedded platforms
    find_package(OpenCV REQUIRED COMPONENTS
            core
            imgproc
            features2d
            calib3d
            imgcodecs
            highgui
    )

    # CUDA for embedded platform acceleration
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        enable_language(CUDA)
        # Embedded platform architectures
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 72 75 87)
        endif()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-extended-lambda")
        message(STATUS "CUDA ${CUDA_VERSION} enabled for architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    # Boost for threading and system utilities
    find_package(Boost REQUIRED COMPONENTS system thread)

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
    )

    if(CUDA_FOUND)
        include_directories(${CUDA_INCLUDE_DIRS})
    endif()
endif()

# Compiler definitions and optimizations
add_definitions(-DCAMERA_STITCHING_MODULE -DPHASE3_BUILD_VALIDATION -DEMBEDDED_PLATFORM)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-O3 -march=native -std=c++14)
    add_compile_options(-ffast-math -funroll-loops)
endif()

# CUDA kernels for image processing acceleration
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

    # CUDA properties for embedded deployment
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

# Camera synchronization library
if(STANDALONE_BUILD)
    set(SYNC_LIB_NAME ${MODULE_NAME}_synchronizer)
else()
    set(SYNC_LIB_NAME lane_fusion_${MODULE_NAME}_synchronizer)
endif()

add_library(${SYNC_LIB_NAME}
        src/camera_synchronizer.cpp
)

if(STANDALONE_BUILD)
    target_link_libraries(${SYNC_LIB_NAME}
            ${ROSCPP_LIBRARY}
            ${ROSTIME_LIBRARY}
            ${OpenCV_LIBRARIES}
            ${Boost_LIBRARIES}
    )
else()
    target_link_libraries(${SYNC_LIB_NAME}
            ${OpenCV_LIBRARIES}
            ${Boost_LIBRARIES}
    )
endif()

# Panoramic stitching library
if(STANDALONE_BUILD)
    set(STITCHER_LIB_NAME ${MODULE_NAME}_stitcher)
else()
    set(STITCHER_LIB_NAME lane_fusion_${MODULE_NAME}_stitcher)
endif()

add_library(${STITCHER_LIB_NAME}
        src/camera_synchronizer.cpp  # Placeholder implementation
)

target_link_libraries(${STITCHER_LIB_NAME}
        ${SYNC_LIB_NAME}
)

if(STANDALONE_BUILD)
    target_link_libraries(${STITCHER_LIB_NAME}
            ${OpenCV_LIBRARIES}
    )
else()
    target_link_libraries(${STITCHER_LIB_NAME}
            ${OpenCV_LIBRARIES}
    )
endif()

if(HAS_CUDA_KERNELS)
    target_link_libraries(${STITCHER_LIB_NAME} ${CUDA_TARGET_NAME})
    target_compile_definitions(${STITCHER_LIB_NAME} PRIVATE CUDA_ACCELERATION)
endif()

# Embedded platform optimizations
set_target_properties(${SYNC_LIB_NAME} PROPERTIES
        COMPILE_FLAGS "-DREAL_TIME_SYNC"
        LINK_FLAGS "-Wl,--gc-sections"
)

set_target_properties(${STITCHER_LIB_NAME} PROPERTIES
        COMPILE_FLAGS "-DREAL_TIME_STITCHING"
        LINK_FLAGS "-Wl,--gc-sections"
)

# Test executable for standalone builds
if(STANDALONE_BUILD)
    add_executable(${MODULE_NAME}_test
            src/camera_stitching_node.cpp
    )

    target_link_libraries(${MODULE_NAME}_test
            ${STITCHER_LIB_NAME}
            ${SYNC_LIB_NAME}
            ${ROSCPP_LIBRARY}
            ${ROSTIME_LIBRARY}
            ${OpenCV_LIBRARIES}
    )

    set_target_properties(${MODULE_NAME}_test PROPERTIES
            COMPILE_FLAGS "-DSTANDALONE_TEST"
    )
endif()

# Export variables for parent project integration
if(NOT STANDALONE_BUILD)
    set(CAMERA_STITCHING_LIBRARIES
            ${SYNC_LIB_NAME}
            ${STITCHER_LIB_NAME}
            PARENT_SCOPE
    )

    if(HAS_CUDA_KERNELS)
        set(CAMERA_STITCHING_CUDA_LIBRARIES
                ${CUDA_TARGET_NAME}
                PARENT_SCOPE
        )
    endif()

    set(CAMERA_STITCHING_INCLUDE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            PARENT_SCOPE
    )
endif()

# Build configuration summary
message(STATUS "=== ${MODULE_NAME} Configuration Summary ===")
message(STATUS "Build mode: ${CMAKE_BUILD_TYPE}")
message(STATUS "Standalone: ${STANDALONE_BUILD}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "CUDA: ${HAS_CUDA_KERNELS}")
if(HAS_CUDA_KERNELS)
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "Libraries: ${SYNC_LIB_NAME}, ${STITCHER_LIB_NAME}")
if(STANDALONE_BUILD)
    message(STATUS "Test executable: ${MODULE_NAME}_test")
endif()
message(STATUS "Phase 3-4 build validation: CONFIGURED")
message(STATUS "===========================================")