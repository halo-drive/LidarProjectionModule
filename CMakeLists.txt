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
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_eigen
  message_filters
  pcl_ros
  pcl_conversions
)

# Find other dependencies
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

# Find CUDA (optional) - suppress warnings
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()
find_package(CUDA QUIET)
if(CUDA_FOUND)
  enable_language(CUDA)
  message(STATUS "CUDA found - enabling GPU acceleration")
  add_definitions(-DUSE_CUDA)
else()
  message(STATUS "CUDA not found - using CPU only")
endif()

# Catkin package setup
catkin_package(
  INCLUDE_DIRS 
    lidar_processing/include
  LIBRARIES 
    lidar_processing
  CATKIN_DEPENDS
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    tf2
    tf2_ros
    tf2_eigen
    message_filters
    pcl_ros
    pcl_conversions
  DEPENDS
    PCL
)

# Include directories
include_directories(
  lidar_processing/include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# Add PCL definitions
add_definitions(${PCL_DEFINITIONS})

# Essential sources only
set(LIDAR_PROCESSING_SOURCES
  lidar_processing/src/point_cloud_proc.cpp
  lidar_processing/src/ground_extraction.cpp
)

# Point cloud processing library
add_library(lidar_processing ${LIDAR_PROCESSING_SOURCES})
target_link_libraries(lidar_processing ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(lidar_processing ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# CUDA libraries (if available)
if(CUDA_FOUND)
  set(CUDA_SOURCES
    lidar_processing/cuda/ground_extraction.cu
  )
  
  # Add CUDA library
  cuda_add_library(cuda_kernels ${CUDA_SOURCES})
  set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
  
  target_link_libraries(lidar_processing cuda_kernels ${CUDA_LIBRARIES})
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

install(TARGETS lidar_ground_extraction_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install directories
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

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

# install(TARGETS
#   lidar_ground_extraction_node
#   calibration_node
#   DESTINATION lib/${PROJECT_NAME}
# )

# # Install directories
# install(DIRECTORY
#   launch/
#   DESTINATION share/${PROJECT_NAME}/launch
# )

# install(DIRECTORY
#   config/
#   DESTINATION share/${PROJECT_NAME}/config
# )

# install(DIRECTORY
#   scripts/
#   DESTINATION share/${PROJECT_NAME}/scripts
#   USE_SOURCE_PERMISSIONS
# )

# # Install headers
# install(DIRECTORY
#   lidar_processing/include/
#   utils/include/
#   calibration/include/
#   DESTINATION include
# )

# # Testing
# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   ament_lint_auto_find_test_dependencies()
  
#   # Unit tests
#   find_package(ament_cmake_gtest REQUIRED)
  
#   # Test point cloud processing
#   ament_add_gtest(test_point_cloud_proc
#     tests/test_point_cloud_proc.cpp
#   )
#   ament_target_dependencies(test_point_cloud_proc
#     rclcpp
#     PCL
#     Eigen3
#   )
#   target_link_libraries(test_point_cloud_proc
#     lidar_processing
#   )
  
#   # Test ground extraction
#   ament_add_gtest(test_ground_extraction
#     tests/test_ground_extraction.cpp
#   )
#   ament_target_dependencies(test_ground_extraction
#     rclcpp
#     PCL
#     Eigen3
#   )
#   target_link_libraries(test_ground_extraction
#     lidar_processing
#   )
  
#   # Test calibration
#   ament_add_gtest(test_calibration
#     tests/test_calibration.cpp
#   )
#   ament_target_dependencies(test_calibration
#     rclcpp
#     PCL
#     Eigen3
#   )
#   target_link_libraries(test_calibration
#     lidar_processing
#     utils
#   )
# endif()

# # Export information
# ament_export_targets(lidar_processing HAS_LIBRARY_TARGET)
# ament_export_targets(utils HAS_LIBRARY_TARGET)
# if(CUDA_FOUND)
#   ament_export_targets(cuda_kernels HAS_LIBRARY_TARGET)
# endif()

# ament_export_dependencies(
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

# ament_export_include_directories(include)

# ament_package()




# OLDER
# cmake_minimum_required(VERSION 3.10)
# project(lane_fusion)

# # Find required packages
# find_package(catkin REQUIRED COMPONENTS
#         roscpp
#         rospy
#         std_msgs
#         sensor_msgs
#         cv_bridge
#         pcl_ros
#         velodyne_msgs
# )

# find_package(OpenCV REQUIRED)
# find_package(PCL REQUIRED)
# find_package(CUDA REQUIRED)
# find_package(Eigen3 REQUIRED)

# # Optional: Find TensorRT
# find_package(TensorRT)

# # CUDA flags
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_87") # Jetson AGX Orin

# # Declare catkin package
# catkin_package(
#         INCLUDE_DIRS include
#         CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs cv_bridge pcl_ros velodyne_msgs
#         DEPENDS OpenCV PCL CUDA Eigen3
# )

# # Include directories
# include_directories(
#         include
#         ${catkin_INCLUDE_DIRS}
#         ${OpenCV_INCLUDE_DIRS}
#         ${PCL_INCLUDE_DIRS}
#         ${CUDA_INCLUDE_DIRS}
#         ${EIGEN3_INCLUDE_DIRS}
# )

# # Add rest of build instructions here
# # ...