
// #ifndef LIDAR_PROCESSING_POINT_TYPES_HPP
// #define LIDAR_PROCESSING_POINT_TYPES_HPP

// #include <pcl/point_types.h>
// #include <pcl/point_cloud.h>
// #include <cstdint>

// namespace lidar_processing {

// // Custom point type for VLP-16 with additional fields
// struct EIGEN_ALIGN16 PointXYZIR {
//     PCL_ADD_POINT4D;                    // x, y, z, padding
//     float intensity;                    // Reflectivity value
//     std::uint16_t ring;                 // Ring number (0-15 for VLP-16)
//     float range;                        // Distance from sensor
//     float timestamp;                    // Point timestamp
//     std::uint8_t sensor_id;             // Sensor identifier (0 for puck, 1 for high-res)
    
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
//     // Default constructor
//     PointXYZIR() : x(0), y(0), z(0), intensity(0), ring(0), range(0), timestamp(0), sensor_id(0) {}
// } EIGEN_ALIGN16;

// // Ground plane point with normal information
// struct EIGEN_ALIGN16 GroundPoint {
//     PCL_ADD_POINT4D;
//     PCL_ADD_NORMAL4D;                   // Normal vector components
//     float curvature;                    // Surface curvature
//     float confidence;                   // Ground classification confidence
    
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
//     // Default constructor
//     GroundPoint() : x(0), y(0), z(0), 
//                    normal_x(0), normal_y(0), normal_z(1), 
//                    curvature(0), confidence(0) {}
// } EIGEN_ALIGN16;

// } // namespace lidar_processing

// // Register custom point fields with PCL
// POINT_CLOUD_REGISTER_POINT_STRUCT(lidar_processing::PointXYZIR,
//     (float, x, x)
//     (float, y, y)
//     (float, z, z)
//     (float, intensity, intensity)
//     (std::uint16_t, ring, ring)
//     (float, range, range)
//     (float, timestamp, timestamp)
//     (std::uint8_t, sensor_id, sensor_id)
// )

// POINT_CLOUD_REGISTER_POINT_STRUCT(lidar_processing::GroundPoint,
//     (float, x, x)
//     (float, y, y)
//     (float, z, z)
//     (float, normal_x, normal_x)
//     (float, normal_y, normal_y)
//     (float, normal_z, normal_z)
//     (float, curvature, curvature)
//     (float, confidence, confidence)
// )

// namespace lidar_processing {

// // Type aliases for convenience
// using PointCloudXYZIR = pcl::PointCloud<PointXYZIR>;
// using PointCloudGround = pcl::PointCloud<GroundPoint>;
// using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
// using PointCloudXYZI = pcl::PointCloud<pcl::PointXYZI>;

// // Sensor configuration constants
// constexpr std::uint8_t SENSOR_VLP16_PUCK = 0;
// constexpr std::uint8_t SENSOR_VLP16_HIGH_RES = 1;
// constexpr int VLP16_RINGS = 16;
// constexpr float VLP16_MIN_RANGE = 0.9f;     // meters
// constexpr float VLP16_MAX_RANGE = 100.0f;   // meters

// } // namespace lidar_processing

// #endif // LIDAR_PROCESSING_POINT_TYPES_HPP

#ifndef LIDAR_PROCESSING_POINT_TYPES_HPP
#define LIDAR_PROCESSING_POINT_TYPES_HPP

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <cstdint>

namespace lidar_processing {

// Use standard PCL point types to avoid linking issues
using PointXYZIR = pcl::PointXYZI;
using PointCloudXYZIR = pcl::PointCloud<pcl::PointXYZI>;
using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudXYZI = pcl::PointCloud<pcl::PointXYZI>;

// For ground points, use PointXYZINormal which is a standard PCL type
using GroundPoint = pcl::PointXYZINormal;
using PointCloudGround = pcl::PointCloud<pcl::PointXYZINormal>;

// Sensor configuration constants
constexpr std::uint8_t SENSOR_VLP16_PUCK = 0;
constexpr std::uint8_t SENSOR_VLP16_HIGH_RES = 1;
constexpr int VLP16_RINGS = 16;
constexpr float VLP16_MIN_RANGE = 0.9f;     // meters
constexpr float VLP16_MAX_RANGE = 100.0f;   // meters

} // namespace lidar_processing

#endif // LIDAR_PROCESSING_POINT_TYPES_HPP