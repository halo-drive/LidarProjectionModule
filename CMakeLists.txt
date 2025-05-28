cmake_minimum_required(VERSION 3.10)
project(lane_fusion)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required ROS packages
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
        velodyne_msgs
        message_filters
        tf2
        tf2_ros
        tf2_geometry_msgs
        tf2_eigen
)

# Find system dependencies
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io features search kdtree segmentation filters)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

# Enable CUDA if available (optional for now)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3; -gencode arch=compute_75,code=sm_75)
endif()

# Optional: Find TensorRT (for Phase 3)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY nvinfer
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
    set(TENSORRT_FOUND TRUE)
    message(STATUS "Found TensorRT: ${TENSORRT_LIBRARY}")
else()
    set(TENSORRT_FOUND FALSE)
    message(WARNING "TensorRT not found - will be needed for Phase 3")
endif()

# Declare catkin package
catkin_package(
        INCLUDE_DIRS 
            lidar_processing/include
        LIBRARIES 
            ${PROJECT_NAME}_lidar_processing
        CATKIN_DEPENDS
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
            velodyne_msgs
            message_filters
            tf2
            tf2_ros
            tf2_geometry_msgs
            tf2_eigen
        DEPENDS
            OpenCV
            PCL
            Boost
)

# Include directories
include_directories(
        lidar_processing/include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

if(TENSORRT_FOUND)
    include_directories(${TENSORRT_INCLUDE_DIR})
endif()

# Add PCL definitions
add_definitions(${PCL_DEFINITIONS})

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

###########
## Build ##
###########

# Point cloud processing library
add_library(${PROJECT_NAME}_lidar_processing
    lidar_processing/src/point_cloud_proc.cpp
    lidar_processing/src/ground_extraction.cpp
)

target_link_libraries(${PROJECT_NAME}_lidar_processing
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
)

# CUDA ground extraction library (if CUDA is available)
if(CUDA_FOUND)
    cuda_add_library(${PROJECT_NAME}_cuda_ground_extraction
        lidar_processing/cuda/ground_extraction.cu
    )
    
    target_link_libraries(${PROJECT_NAME}_cuda_ground_extraction
        ${CUDA_LIBRARIES}
    )
    
    # Link CUDA library to main library
    target_link_libraries(${PROJECT_NAME}_lidar_processing
        ${PROJECT_NAME}_cuda_ground_extraction
    )
endif()

# Ground extraction node executable (YOUR EXISTING NODE)
add_executable(lidar_ground_extraction_node
    lidar_processing/src/lidar_ground_extraction_node.cpp
)

target_link_libraries(lidar_ground_extraction_node
    ${PROJECT_NAME}_lidar_processing
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
)

# Add dependencies
add_dependencies(${PROJECT_NAME}_lidar_processing 
    ${${PROJECT_NAME}_EXPORTED_TARGETS} 
    ${catkin_EXPORTED_TARGETS}
)

add_dependencies(lidar_ground_extraction_node 
    ${${PROJECT_NAME}_EXPORTED_TARGETS} 
    ${catkin_EXPORTED_TARGETS}
)

#############
## Install ##
#############

# Install executables
install(TARGETS 
        lidar_ground_extraction_node
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install libraries
install(TARGETS 
        ${PROJECT_NAME}_lidar_processing
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

# Install header files
install(DIRECTORY lidar_processing/include/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.hpp"
)

# Install launch files
install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
        FILES_MATCHING PATTERN "*.launch" PATTERN "*.py"
        PATTERN ".svn" EXCLUDE)

# Install config files
install(DIRECTORY config/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
        PATTERN ".svn" EXCLUDE)

# Install scripts
install(PROGRAMS
        scripts/setup_environment.sh
        scripts/build.sh
        scripts/calibration_collection.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Build status message
message(STATUS "=== LidarProjectionLane Build Configuration ===")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "PCL: Found")
if(CUDA_FOUND)
    message(STATUS "CUDA: ${CUDA_VERSION}")
else()
    message(STATUS "CUDA: Not found (optional)")
endif()
message(STATUS "Eigen3: ${EIGEN3_VERSION}")
if(TENSORRT_FOUND)
    message(STATUS "TensorRT: Found")
else()
    message(STATUS "TensorRT: Not found (needed for Phase 3)")
endif()
message(STATUS "Build target: lidar_ground_extraction_node")
message(STATUS "==============================================")

# cmake_minimum_required(VERSION 3.10)
# project(lane_fusion)

# # Set C++ standard
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # Enable CUDA language
# enable_language(CUDA)

# # Find required ROS packages
# find_package(catkin REQUIRED COMPONENTS
#         roscpp
#         rospy
#         std_msgs
#         sensor_msgs
#         geometry_msgs
#         cv_bridge
#         image_transport
#         pcl_ros
#         velodyne_msgs
#         message_filters
#         tf2
#         tf2_ros
#         tf2_geometry_msgs
# )

# # Find system dependencies
# find_package(OpenCV REQUIRED)
# find_package(PCL REQUIRED COMPONENTS common io features search kdtree)
# find_package(CUDA REQUIRED)
# find_package(Eigen3 REQUIRED)

# # Optional: Find TensorRT (for Phase 3)
# find_path(TENSORRT_INCLUDE_DIR NvInfer.h
#         HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
#         PATH_SUFFIXES include)
# find_library(TENSORRT_LIBRARY nvinfer
#         HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
#         PATH_SUFFIXES lib lib64 lib/x64)

# if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
#     set(TENSORRT_FOUND TRUE)
#     message(STATUS "Found TensorRT: ${TENSORRT_LIBRARY}")
# else()
#     set(TENSORRT_FOUND FALSE)
#     message(WARNING "TensorRT not found - will be needed for Phase 3")
# endif()

# # CUDA compilation flags for x86 platform
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3; -gencode arch=compute_75,code=sm_75)

# # Declare catkin package (minimal for Phase 1-2)
# catkin_package(
#         CATKIN_DEPENDS
#         roscpp
#         rospy
#         std_msgs
#         sensor_msgs
#         geometry_msgs
#         cv_bridge
#         image_transport
#         pcl_ros
#         velodyne_msgs
#         message_filters
#         tf2
#         tf2_ros
#         tf2_geometry_msgs
#         DEPENDS
#         OpenCV
#         PCL
#         CUDA
# )

# # Include directories (only existing ones)
# include_directories(
#         ${catkin_INCLUDE_DIRS}
#         ${OpenCV_INCLUDE_DIRS}
#         ${PCL_INCLUDE_DIRS}
#         ${CUDA_INCLUDE_DIRS}
#         ${EIGEN3_INCLUDE_DIRS}
# )

# if(TENSORRT_FOUND)
#     include_directories(${TENSORRT_INCLUDE_DIR})
# endif()

# # Compiler-specific options
# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     add_compile_options(-Wall -Wextra -Wpedantic -O3)
# endif()

# # Phase 3+ will add subdirectories and libraries here
# # For now, just ensure the build system works

# # Install launch files (corrected syntax)
# install(DIRECTORY launch/
#         DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
#         FILES_MATCHING PATTERN "*.launch" PATTERN "*.py"
#         PATTERN ".svn" EXCLUDE)

# # Install config files
# install(DIRECTORY config/
#         DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
#         PATTERN ".svn" EXCLUDE)

# # Install scripts
# install(PROGRAMS
#         scripts/setup_environment.sh
#         scripts/build.sh
#         scripts/calibration_collection.py
#         DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# # Testing (optional)
# if(CATKIN_ENABLE_TESTING)
#     find_package(rostest REQUIRED)
#     # Phase 3+ will add tests here
# endif()

# # Build status message
# message(STATUS "=== LidarProjectionLane Build Configuration ===")
# message(STATUS "OpenCV: ${OpenCV_VERSION}")
# message(STATUS "PCL: Found")
# message(STATUS "CUDA: ${CUDA_VERSION}")
# message(STATUS "Eigen3: ${EIGEN3_VERSION}")
# if(TENSORRT_FOUND)
#     message(STATUS "TensorRT: Found")
# else()
#     message(STATUS "TensorRT: Not found (needed for Phase 3)")
# endif()
# message(STATUS "==============================================")