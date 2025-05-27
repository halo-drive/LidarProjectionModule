#ifndef LIDAR_PROCESSING_POINT_TYPES_HPP
#define LIDAR_PROCESSING_POINT_TYPES_HPP

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace lidar_processing {

// Custom point type for VLP-16 with additional fields
struct EIGEN_ALIGN16 PointXYZIR {
    PCL_ADD_POINT4D;                    // x, y, z, padding
    float intensity;                    // Reflectivity value
    uint16_t ring;                      // Ring number (0-15 for VLP-16)
    float range;                        // Distance from sensor
    float timestamp;                    // Point timestamp
    uint8_t sensor_id;                  // Sensor identifier (0 for puck, 1 for high-res)
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

// Ground plane point with normal information
struct EIGEN_ALIGN16 GroundPoint {
    PCL_ADD_POINT4D;
    PCL_ADD_NORMAL4D;                   // Normal vector components
    float curvature;                    // Surface curvature
    float confidence;                   // Ground classification confidence
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

} // namespace lidar_processing

// Register point types with PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(lidar_processing::PointXYZIR,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint16_t, ring, ring)
    (float, range, range)
    (float, timestamp, timestamp)
    (uint8_t, sensor_id, sensor_id)
)

POINT_CLOUD_REGISTER_POINT_STRUCT(lidar_processing::GroundPoint,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, normal_x, normal_x)
    (float, normal_y, normal_y)
    (float, normal_z, normal_z)
    (float, curvature, curvature)
    (float, confidence, confidence)
)

namespace lidar_processing {

// Type aliases for convenience
using PointCloudXYZIR = pcl::PointCloud<PointXYZIR>;
using PointCloudGround = pcl::PointCloud<GroundPoint>;
using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudXYZI = pcl::PointCloud<pcl::PointXYZI>;

// Sensor configuration constants
constexpr uint8_t SENSOR_VLP16_PUCK = 0;
constexpr uint8_t SENSOR_VLP16_HIGH_RES = 1;
constexpr int VLP16_RINGS = 16;
constexpr float VLP16_MIN_RANGE = 0.9f;     // meters
constexpr float VLP16_MAX_RANGE = 100.0f;   // meters

} // namespace lidar_processing

#endif // LIDAR_PROCESSING_POINT_TYPES_HPP