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