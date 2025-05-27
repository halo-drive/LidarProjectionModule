cmake_minimum_required(VERSION 3.10)
project(lidar_projection_lane)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Find ROS1 packages
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        sensor_msgs
        geometry_msgs
        cv_bridge
        image_transport
        pcl_ros
        velodyne_msgs
        message_filters
        tf2
        tf2_ros
        tf2_geometry_msgs
)

# Find system dependencies
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io features search kdtree)
find_package(CUDA REQUIRED)
# Find other dependencies
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

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

# CUDA compilation flags for x86 platform
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3; -gencode arch=compute_75,code=sm_75)

# Declare catkin package (minimal for Phase 1-2)
catkin_package(
        CATKIN_DEPENDS
        roscpp
        rospy
        std_msgs
        sensor_msgs
        geometry_msgs
        cv_bridge
        image_transport
        pcl_ros
        velodyne_msgs
        message_filters
        tf2
        tf2_ros
        tf2_geometry_msgs
        DEPENDS
        OpenCV
        PCL
        CUDA
)

# Include directories (only existing ones)
include_directories(
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
)

if(TENSORRT_FOUND)
    include_directories(${TENSORRT_INCLUDE_DIR})
endif()

# Main executable - ROS1 version
add_executable(lidar_ground_extraction_node
  lidar_processing/src/lidar_ground_extraction_node.cpp
)

target_link_libraries(lidar_ground_extraction_node
  lidar_processing
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_dependencies(lidar_ground_extraction_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# Install targets
install(TARGETS lidar_processing
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CUDA_FOUND)
  install(TARGETS cuda_kernels
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
endif()

# Phase 3+ will add subdirectories and libraries here
# For now, just ensure the build system works

# Install launch files (corrected syntax)
install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
        FILES_MATCHING PATTERN "*.launch" PATTERN "*.py"
        PATTERN ".svn" EXCLUDE)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

# Install headers
install(DIRECTORY
  lidar_processing/include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

# ros2
# cmake_minimum_required(VERSION 3.16)
# project(lidar_projection_lane)

# # Set C++ standard
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # Compiler options
# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic -O3)
# endif()

# # Find dependencies
# find_package(ament_cmake REQUIRED)
# find_package(rclcpp REQUIRED)
# find_package(sensor_msgs REQUIRED)
# find_package(geometry_msgs REQUIRED)
# find_package(visualization_msgs REQUIRED)
# find_package(std_msgs REQUIRED)
# find_package(tf2 REQUIRED)
# find_package(tf2_ros REQUIRED)
# find_package(tf2_eigen REQUIRED)
# find_package(message_filters REQUIRED)
# find_package(PCL REQUIRED)
# find_package(pcl_conversions REQUIRED)
# find_package(Eigen3 REQUIRED)

# # Find CUDA (optional)
# find_package(CUDA QUIET)
# if(CUDA_FOUND)
#   enable_language(CUDA)
#   message(STATUS "CUDA found - enabling GPU acceleration")
#   add_definitions(-DUSE_CUDA)
# else()
#   message(STATUS "CUDA not found - using CPU only")
# endif()

# # Include directories
# include_directories(
#   ${CMAKE_CURRENT_SOURCE_DIR}
#   ${PCL_INCLUDE_DIRS}
#   ${EIGEN3_INCLUDE_DIRS}
# )

# # Add PCL definitions
# add_definitions(${PCL_DEFINITIONS})

# # Create libraries
# set(LIDAR_PROCESSING_SOURCES
#   lidar_processing/src/point_cloud_proc.cpp
#   lidar_processing/src/ground_extraction.cpp
# )

# set(UTILS_SOURCES
#   utils/src/time_sync.cpp
#   utils/src/memory_management.cpp
#   utils/src/visualization.cpp
# )

# # Point cloud processing library
# add_library(lidar_processing SHARED ${LIDAR_PROCESSING_SOURCES})
# ament_target_dependencies(lidar_processing
#   rclcpp
#   sensor_msgs
#   PCL
#   pcl_conversions
#   Eigen3
# )
# target_link_libraries(lidar_processing ${PCL_LIBRARIES})

# # Utils library
# add_library(utils SHARED ${UTILS_SOURCES})
# ament_target_dependencies(utils
#   rclcpp
#   sensor_msgs
#   geometry_msgs
#   tf2
#   tf2_ros
#   tf2_eigen
# )

# # CUDA libraries (if available)
# if(CUDA_FOUND)
#   set(CUDA_SOURCES
#     lidar_processing/cuda/ground_extraction.cu
#     lidar_processing/cuda/voxel_grid.cu
#   )
  
#   # Set CUDA properties
#   set_property(TARGET lidar_processing PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  
#   # Add CUDA library
#   add_library(cuda_kernels SHARED ${CUDA_SOURCES})
#   set_target_properties(cuda_kernels PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES "72;75;86;89"  # Jetson and modern GPU architectures
#   )
  
#   target_link_libraries(lidar_processing cuda_kernels ${CUDA_LIBRARIES})
# endif()

# # Main executable
# add_executable(lidar_ground_extraction_node
#   lidar_processing/src/lidar_ground_extraction_node.cpp
# )

# ament_target_dependencies(lidar_ground_extraction_node
#   rclcpp
#   sensor_msgs
#   geometry_msgs
#   visualization_msgs
#   std_msgs
#   tf2
#   tf2_ros
#   tf2_eigen
#   message_filters
#   PCL
#   pcl_conversions
#   Eigen3
# )

# target_link_libraries(lidar_ground_extraction_node
#   lidar_processing
#   utils
#   ${PCL_LIBRARIES}
# )

# # Calibration tools
# add_executable(calibration_node
#   calibration/tools/calibration_node.cpp
#   calibration/src/camera_calibrator.cpp
#   calibration/src/lidar_calibrator.cpp
#   calibration/src/extrinsic_calibrator.cpp
# )

# ament_target_dependencies(calibration_node
#   rclcpp
#   sensor_msgs
#   geometry_msgs
#   tf2
#   tf2_ros
#   PCL
#   pcl_conversions
#   Eigen3
# )

# target_link_libraries(calibration_node
#   utils
#   ${PCL_LIBRARIES}
# )

# # Install targets
# install(TARGETS
#   lidar_processing
#   utils
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib
#   RUNTIME DESTINATION bin
# )

# if(CUDA_FOUND)
#   install(TARGETS cuda_kernels
#     ARCHIVE DESTINATION lib
#     LIBRARY DESTINATION lib
#     RUNTIME DESTINATION bin
#   )
# endif()

# Testing (optional)
if(CATKIN_ENABLE_TESTING)
    find_package(rostest REQUIRED)
    # Phase 3+ will add tests here
endif()

# Build status message
message(STATUS "=== LidarProjectionLane Build Configuration ===")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "PCL: Found")
message(STATUS "CUDA: ${CUDA_VERSION}")
message(STATUS "Eigen3: ${EIGEN3_VERSION}")
if(TENSORRT_FOUND)
    message(STATUS "TensorRT: Found")
else()
    message(STATUS "TensorRT: Not found (needed for Phase 3)")
endif()
message(STATUS "==============================================")