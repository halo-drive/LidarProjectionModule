# Isolated build configuration for sensor_fusion module (projection-based LiDAR-camera fusion)
cmake_minimum_required(VERSION 3.0.2)

# Set module name for multi-modal sensor fusion
set(MODULE_NAME "sensor_fusion")
message(STATUS "Building ${MODULE_NAME} module for projection-based LiDAR-camera integration")

# Check if building as part of main package or standalone
if(NOT DEFINED PROJECT_NAME OR PROJECT_NAME STREQUAL "")
    project(${MODULE_NAME}_standalone)
    set(STANDALONE_BUILD TRUE)

    # Find catkin for standalone build
    find_package(catkin REQUIRED COMPONENTS
            roscpp
            rospy
            std_msgs
            sensor_msgs
            geometry_msgs
            visualization_msgs
            cv_bridge
            image_transport
            pcl_ros
            pcl_conversions
            message_filters
            tf2
            tf2_ros
            tf2_geometry_msgs
            tf2_eigen
            camera_info_manager
    )

    # System dependencies for fusion algorithms
    find_package(OpenCV REQUIRED)
    find_package(PCL REQUIRED COMPONENTS common io features search kdtree)
    find_package(Eigen3 REQUIRED)
    find_package(Boost REQUIRED COMPONENTS system thread)

    # CUDA for GPU-accelerated projection and fusion operations
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        enable_language(CUDA)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 72 75 87)
        endif()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-extended-lambda")
        message(STATUS "CUDA ${CUDA_VERSION} enabled for projection-based fusion acceleration")
    endif()

    # Standalone catkin package configuration
    catkin_package(
            INCLUDE_DIRS include
            LIBRARIES
            ${MODULE_NAME}_projection
            ${MODULE_NAME}_lane_model
            ${MODULE_NAME}_calibration
            ${MODULE_NAME}_cuda_kernels
            CATKIN_DEPENDS
            roscpp
            std_msgs
            sensor_msgs
            cv_bridge
            pcl_ros
            message_filters
            DEPENDS
            OpenCV
            PCL
            Eigen3
            Boost
    )
else()
    set(STANDALONE_BUILD FALSE)
    message(STATUS "Building ${MODULE_NAME} as part of ${PROJECT_NAME} package")
endif()

# Module-specific include directories
include_directories(
        include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(STANDALONE_BUILD)
    include_directories(
            ${catkin_INCLUDE_DIRS}
            ${OpenCV_INCLUDE_DIRS}
            ${PCL_INCLUDE_DIRS}
            ${EIGEN3_INCLUDE_DIRS}
            ${Boost_INCLUDE_DIRS}
    )

    if(CUDA_FOUND)
        include_directories(${CUDA_INCLUDE_DIRS})
    endif()

    add_definitions(${PCL_DEFINITIONS})
endif()

# Cross-module dependencies for integrated build
if(NOT STANDALONE_BUILD)
    # Include directories from other modules
    include_directories(
            ${CMAKE_CURRENT_SOURCE_DIR}/../lidar_processing/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../lane_detection/include
    )
endif()

# Embedded platform and fusion-specific definitions
add_definitions(-DEMBEDDED_PLATFORM -DSENSOR_FUSION_MODULE -DPROJECTION_BASED_FUSION)

# High-performance computing optimizations for real-time fusion
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-O3 -march=native -mtune=native -funroll-loops)
    add_compile_options(-ffast-math -finline-functions -fprefetch-loop-arrays)
    add_compile_options(-fopenmp)  # OpenMP for parallel processing
endif()

# CUDA flags optimized for geometric projection operations
if(CUDA_FOUND OR (DEFINED WORKSPACE_CUDA_FOUND AND WORKSPACE_CUDA_FOUND))
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-extended-lambda")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --maxrregcount=48")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3 --ptxas-options=-allow-expensive-optimizations=true")
endif()

## CUDA Projection and Fusion Kernels
if(CUDA_FOUND OR (DEFINED WORKSPACE_CUDA_FOUND AND WORKSPACE_CUDA_FOUND))
    if(STANDALONE_BUILD)
        cuda_add_library(${MODULE_NAME}_cuda_kernels
                cuda/projection_kernels.cu
                cuda/lane_fitting_kernels.cu
                cuda/correspondence_kernels.cu
                cuda/geometric_validation.cu
                cuda/ransac_kernels.cu
        )
        set(CUDA_TARGET_NAME ${MODULE_NAME}_cuda_kernels)
    else()
        cuda_add_library(lane_fusion_${MODULE_NAME}_cuda_kernels
                ${CMAKE_CURRENT_SOURCE_DIR}/cuda/projection_kernels.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/cuda/lane_fitting_kernels.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/cuda/correspondence_kernels.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/cuda/geometric_validation.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ransac_kernels.cu
        )
        set(CUDA_TARGET_NAME lane_fusion_${MODULE_NAME}_cuda_kernels)
    endif()

    if(STANDALONE_BUILD)
        target_link_libraries(${CUDA_TARGET_NAME}
                ${CUDA_LIBRARIES}
                ${CUDA_CUBLAS_LIBRARIES}
                ${catkin_LIBRARIES}
        )
    else()
        target_link_libraries(${CUDA_TARGET_NAME}
                ${CUDA_LIBRARIES}
                ${CUDA_CUBLAS_LIBRARIES}
        )
    endif()

    # CUDA properties optimized for geometric computations
    set_target_properties(${CUDA_TARGET_NAME} PROPERTIES
            CUDA_RUNTIME_LIBRARY Static
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            POSITION_INDEPENDENT_CODE ON
            CUDA_SEPARABLE_COMPILATION ON
    )

    if(DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(${CUDA_TARGET_NAME} PROPERTIES
                CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
    endif()

    set(HAS_CUDA_KERNELS TRUE)
else()
    set(HAS_CUDA_KERNELS FALSE)
    message(STATUS "${MODULE_NAME}: Building without CUDA acceleration")
endif()

## Projection and Coordinate Transformation Library
if(STANDALONE_BUILD)
    set(PROJECTION_LIB_NAME ${MODULE_NAME}_projection)
else()
    set(PROJECTION_LIB_NAME lane_fusion_${MODULE_NAME}_projection)
endif()

add_library(${PROJECTION_LIB_NAME}
        src/projection.cpp
        src/camera_lidar_calibration.cpp
        src/coordinate_transforms.cpp
        src/geometric_validator.cpp
        src/temporal_synchronizer.cpp
)

if(STANDALONE_BUILD)
    target_link_libraries(${PROJECTION_LIB_NAME}
            ${catkin_LIBRARIES}
            ${OpenCV_LIBRARIES}
            ${PCL_LIBRARIES}
            ${Eigen3_LIBRARIES}
    )
else()
    target_link_libraries(${PROJECTION_LIB_NAME}
            ${OpenCV_LIBRARIES}
            ${PCL_LIBRARIES}
            ${Eigen3_LIBRARIES}
    )
endif()

if(HAS_CUDA_KERNELS)
    target_link_libraries(${PROJECTION_LIB_NAME} ${CUDA_TARGET_NAME})
    target_compile_definitions(${PROJECTION_LIB_NAME} PRIVATE CUDA_PROJECTION_ACCELERATION)
endif()

# Real-time projection optimizations
set_target_properties(${PROJECTION_LIB_NAME} PROPERTIES
        COMPILE_FLAGS "-DREAL_TIME_PROJECTION -DGEOMETRIC_CONSISTENCY_VALIDATION"
)

## 3D Lane Model Fitting and RANSAC Library
if(STANDALONE_BUILD)
    set(LANE_MODEL_LIB_NAME ${MODULE_NAME}_lane_model)
else()
    set(LANE_MODEL_LIB_NAME lane_fusion_${MODULE_NAME}_lane_model)
endif()

add_library(${LANE_MODEL_LIB_NAME}
        src/lane_model.cpp
        src/ransac_fitting.cpp
        src/polynomial_solver.cpp
        src/curve_optimizer.cpp
        src/confidence_estimator.cpp
)

target_link_libraries(${LANE_MODEL_LIB_NAME}
        ${PROJECTION_LIB_NAME}
)

if(STANDALONE_BUILD)
    target_link_libraries(${LANE_MODEL_LIB_NAME}
            ${catkin_LIBRARIES}
            ${OpenCV_LIBRARIES}
            ${Eigen3_LIBRARIES}
    )
else()
    target_link_libraries(${LANE_MODEL_LIB_NAME}
            ${OpenCV_LIBRARIES}
            ${Eigen3_LIBRARIES}
    )
endif()

if(HAS_CUDA_KERNELS)
    target_link_libraries(${LANE_MODEL_LIB_NAME} ${CUDA_TARGET_NAME})
    target_compile_definitions(${LANE_MODEL_LIB_NAME} PRIVATE CUDA_RANSAC_ACCELERATION)
endif()

# RANSAC and curve fitting optimizations
set_target_properties(${LANE_MODEL_LIB_NAME} PROPERTIES
        COMPILE_FLAGS "-DROBUST_ESTIMATION -DPARALLEL_RANSAC -DOUTLIER_REJECTION"
)

## Sensor Calibration and Extrinsic Parameter Management
if(STANDALONE_BUILD)
    set(CALIBRATION_LIB_NAME ${MODULE_NAME}_calibration)
else()
    set(CALIBRATION_LIB_NAME lane_fusion_${MODULE_NAME}_calibration)
endif()

add_library(${CALIBRATION_LIB_NAME}
        src/extrinsic_calibrator.cpp
        src/camera_calibration.cpp
        src/lidar_calibration.cpp
        src/automatic_calibration.cpp
        src/calibration_validator.cpp
)

target_link_libraries(${CALIBRATION_LIB_NAME}
        ${PROJECTION_LIB_NAME}
)

if(STANDALONE_BUILD)
    target_link_libraries(${CALIBRATION_LIB_NAME}
            ${catkin_LIBRARIES}
            ${OpenCV_LIBRARIES}
            ${PCL_LIBRARIES}
            ${Eigen3_LIBRARIES}
    )
else()
    target_link_libraries(${CALIBRATION_LIB_NAME}
            ${OpenCV_LIBRARIES}
            ${PCL_LIBRARIES}
            ${Eigen3_LIBRARIES}
    )
endif()

# Calibration accuracy and precision optimizations
set_target_properties(${CALIBRATION_LIB_NAME} PROPERTIES
        COMPILE_FLAGS "-DHIGH_PRECISION_CALIBRATION -DITERACTIVE_REFINEMENT"
)

## Main Fusion Node Library
if(STANDALONE_BUILD)
    set(FUSION_NODE_LIB_NAME ${MODULE_NAME}_node)
else()
    set(FUSION_NODE_LIB_NAME lane_fusion_${MODULE_NAME}_node)
endif()

add_library(${FUSION_NODE_LIB_NAME}
        src/lane_fusion_node.cpp
        src/multi_sensor_synchronizer.cpp
        src/fusion_orchestrator.cpp
        src/output_generator.cpp
)

target_link_libraries(${FUSION_NODE_LIB_NAME}
        ${PROJECTION_LIB_NAME}
        ${LANE_MODEL_LIB_NAME}
        ${CALIBRATION_LIB_NAME}
)

if(STANDALONE_BUILD)
    target_link_libraries(${FUSION_NODE_LIB_NAME}
            ${catkin_LIBRARIES}
            ${OpenCV_LIBRARIES}
            ${PCL_LIBRARIES}
    )
else()
    target_link_libraries(${FUSION_NODE_LIB_NAME}
            ${OpenCV_LIBRARIES}
            ${PCL_LIBRARIES}
    )

    # Link against other module libraries in integrated build
    if(DEFINED LIDAR_PROCESSING_LIBRARIES)
        target_link_libraries(${FUSION_NODE_LIB_NAME} ${LIDAR_PROCESSING_LIBRARIES})
    endif()

    if(DEFINED LANE_DETECTION_LIBRARIES)
        target_link_libraries(${FUSION_NODE_LIB_NAME} ${LANE_DETECTION_LIBRARIES})
    endif()
endif()

# Real-time fusion orchestration optimizations
set_target_properties(${FUSION_NODE_LIB_NAME} PROPERTIES
        COMPILE_FLAGS "-DREAL_TIME_FUSION -DMULTI_SENSOR_ORCHESTRATION -DLOW_LATENCY_PROCESSING"
)

## Executables (only for standalone build)
if(STANDALONE_BUILD)
    # Main sensor fusion node
    add_executable(${MODULE_NAME}_fusion_node
            src/nodes/sensor_fusion_node.cpp
    )

    target_link_libraries(${MODULE_NAME}_fusion_node
            ${FUSION_NODE_LIB_NAME}
            ${PROJECTION_LIB_NAME}
            ${LANE_MODEL_LIB_NAME}
            ${CALIBRATION_LIB_NAME}
            ${catkin_LIBRARIES}
            ${OpenCV_LIBRARIES}
            ${PCL_LIBRARIES}
    )

    # Calibration tool for camera-LiDAR extrinsic parameter estimation
    add_executable(${MODULE_NAME}_calibration_tool
            src/tools/calibration_tool.cpp
    )

    target_link_libraries(${MODULE_NAME}_calibration_tool
            ${CALIBRATION_LIB_NAME}
            ${PROJECTION_LIB_NAME}
            ${catkin_LIBRARIES}
            ${OpenCV_LIBRARIES}
            ${PCL_LIBRARIES}
    )

    # Fusion performance analyzer and validator
    add_executable(${MODULE_NAME}_performance_analyzer
            src/tools/fusion_analyzer.cpp
    )

    target_link_libraries(${MODULE_NAME}_performance_analyzer
            ${FUSION_NODE_LIB_NAME}
            ${LANE_MODEL_LIB_NAME}
            ${catkin_LIBRARIES}
            ${OpenCV_LIBRARIES}
    )

    # Real-time scheduling for fusion node
    set_target_properties(${MODULE_NAME}_fusion_node PROPERTIES
            COMPILE_FLAGS "-DREAL_TIME_SCHEDULING -pthread -DFUSION_PRIORITY_SCHEDULING"
            LINK_FLAGS "-pthread -lrt -Wl,--no-as-needed"
    )
endif()

## Dependencies
if(STANDALONE_BUILD)
    add_dependencies(${PROJECTION_LIB_NAME}
            ${${PROJECT_NAME}_EXPORTED_TARGETS}
            ${catkin_EXPORTED_TARGETS}
    )

    add_dependencies(${LANE_MODEL_LIB_NAME}
            ${${PROJECT_NAME}_EXPORTED_TARGETS}
            ${catkin_EXPORTED_TARGETS}
    )

    add_dependencies(${CALIBRATION_LIB_NAME}
            ${${PROJECT_NAME}_EXPORTED_TARGETS}
            ${catkin_EXPORTED_TARGETS}
    )

    if(HAS_CUDA_KERNELS)
        add_dependencies(${CUDA_TARGET_NAME}
                ${${PROJECT_NAME}_EXPORTED_TARGETS}
                ${catkin_EXPORTED_TARGETS}
        )
    endif()
endif()

## Export library names for parent project
if(NOT STANDALONE_BUILD)
    set(SENSOR_FUSION_LIBRARIES
            ${PROJECTION_LIB_NAME}
            ${LANE_MODEL_LIB_NAME}
            ${CALIBRATION_LIB_NAME}
            ${FUSION_NODE_LIB_NAME}
            PARENT_SCOPE
    )

    if(HAS_CUDA_KERNELS)
        set(SENSOR_FUSION_CUDA_LIBRARIES
                ${CUDA_TARGET_NAME}
                PARENT_SCOPE
        )
    endif()

    set(SENSOR_FUSION_INCLUDE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            PARENT_SCOPE
    )
endif()

## Module build summary with fusion-specific information
message(STATUS "=== ${MODULE_NAME} Module Build Configuration ===")
message(STATUS "Standalone build: ${STANDALONE_BUILD}")
message(STATUS "Projection-based fusion: ENABLED")
message(STATUS "CUDA acceleration: ${HAS_CUDA_KERNELS}")
message(STATUS "Real-time constraints: ENABLED")
message(STATUS "Multi-sensor synchronization: ENABLED")
message(STATUS "RANSAC-based 3D fitting: ENABLED")
message(STATUS "Geometric consistency validation: ENABLED")
message(STATUS "Target libraries: ${PROJECTION_LIB_NAME}, ${LANE_MODEL_LIB_NAME}")
message(STATUS "Calibration framework: ${CALIBRATION_LIB_NAME}")
if(HAS_CUDA_KERNELS)
    message(STATUS "CUDA kernels: ${CUDA_TARGET_NAME}")
endif()
message(STATUS "Cross-modal validation: Point-to-pixel correspondence")
message(STATUS "=====================================================")