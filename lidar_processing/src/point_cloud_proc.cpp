#include "lidar_processing/point_cloud_proc.hpp"
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <chrono>
#include <cmath>

namespace lidar_processing {

PointCloudProcessor::PointCloudProcessor(const FilterParams& params)
    : params_(params) {
    
    // Configure PCL filters
    voxel_filter_.setLeafSize(params_.voxel_size, params_.voxel_size, params_.voxel_size);
    
    statistical_filter_.setMeanK(params_.statistical_k);
    statistical_filter_.setStddevMulThresh(params_.statistical_std_mul);
    
    radius_filter_.setRadiusSearch(params_.radius_search);
    radius_filter_.setMinNeighborsInRadius(params_.min_neighbors);
}

PointCloudXYZIR::Ptr PointCloudProcessor::mergePointClouds(
    const PointCloudXYZI::ConstPtr& cloud1,
    const PointCloudXYZI::ConstPtr& cloud2,
    const Eigen::Matrix4f& transform2) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Reset statistics
    last_stats_ = ProcessingStats{};
    last_stats_.input_points = cloud1->size() + cloud2->size();
    
    auto merged_cloud = std::make_shared<PointCloudXYZIR>();
    
    // Reserve memory for efficiency
    merged_cloud->reserve(cloud1->size() + cloud2->size());
    
    // Process first sensor (no transformation needed)
    auto cloud1_converted = transformPointCloud(cloud1, Eigen::Matrix4f::Identity(), SENSOR_VLP16_PUCK);
    
    // Process second sensor (apply transformation)
    auto cloud2_converted = transformPointCloud(cloud2, transform2, SENSOR_VLP16_HIGH_RES);
    
    // Merge the clouds
    *merged_cloud = *cloud1_converted;
    *merged_cloud += *cloud2_converted;
    
    // Set header information
    merged_cloud->header = cloud1->header;
    merged_cloud->is_dense = false; // May contain NaN/Inf values
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.merge_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    last_stats_.output_points = merged_cloud->size();
    
    RCLCPP_INFO(rclcpp::get_logger("PointCloudProcessor"),
        "Merged %zu + %zu points into %zu points in %.2f ms",
        cloud1->size(), cloud2->size(), merged_cloud->size(), last_stats_.merge_time_ms);
    
    return merged_cloud;
}

PointCloudXYZIR::Ptr PointCloudProcessor::convertFromROS(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg,
    uint8_t sensor_id) {
    
    // Convert ROS message to PCL
    auto pcl_cloud = std::make_shared<PointCloudXYZI>();
    pcl::fromROSMsg(*msg, *pcl_cloud);
    
    // Transform to identity (no transformation for single sensor conversion)
    return transformPointCloud(pcl_cloud, Eigen::Matrix4f::Identity(), sensor_id);
}

PointCloudXYZIR::Ptr PointCloudProcessor::applyFilters(const PointCloudXYZIR::ConstPtr& input) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    RCLCPP_DEBUG(rclcpp::get_logger("PointCloudProcessor"),
        "Applying filters to %zu points", input->size());
    
    // Apply filters in sequence
    auto filtered = rangeFilter(input);
    filtered = voxelGridFilter(filtered);
    filtered = statisticalOutlierFilter(filtered);
    filtered = radiusOutlierFilter(filtered);
    filtered = organizeByRings(filtered);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.filter_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    RCLCPP_INFO(rclcpp::get_logger("PointCloudProcessor"),
        "Filtered %zu -> %zu points in %.2f ms",
        input->size(), filtered->size(), last_stats_.filter_time_ms);
    
    return filtered;
}

PointCloudXYZIR::Ptr PointCloudProcessor::rangeFilter(const PointCloudXYZIR::ConstPtr& input) {
    auto filtered = std::make_shared<PointCloudXYZIR>();
    filtered->reserve(input->size());
    
    for (const auto& point : input->points) {
        // Skip invalid points
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            continue;
        }
        
        // Check range constraints
        if (point.range >= params_.min_range && point.range <= params_.max_range &&
            point.z >= params_.min_height && point.z <= params_.max_height) {
            filtered->push_back(point);
        }
    }
    
    filtered->header = input->header;
    filtered->is_dense = true;
    
    return filtered;
}

PointCloudXYZIR::Ptr PointCloudProcessor::voxelGridFilter(const PointCloudXYZIR::ConstPtr& input) {
    auto filtered = std::make_shared<PointCloudXYZIR>();
    
    voxel_filter_.setInputCloud(input);
    voxel_filter_.filter(*filtered);
    
    return filtered;
}

PointCloudXYZIR::Ptr PointCloudProcessor::statisticalOutlierFilter(const PointCloudXYZIR::ConstPtr& input) {
    auto filtered = std::make_shared<PointCloudXYZIR>();
    
    statistical_filter_.setInputCloud(input);
    statistical_filter_.filter(*filtered);
    
    return filtered;
}

PointCloudXYZIR::Ptr PointCloudProcessor::radiusOutlierFilter(const PointCloudXYZIR::ConstPtr& input) {
    auto filtered = std::make_shared<PointCloudXYZIR>();
    
    radius_filter_.setInputCloud(input);
    radius_filter_.filter(*filtered);
    
    return filtered;
}

PointCloudXYZIR::Ptr PointCloudProcessor::organizeByRings(const PointCloudXYZIR::ConstPtr& input) {
    auto organized = std::make_shared<PointCloudXYZIR>(*input);
    
    // Sort points by ring and azimuth for better organization
    std::sort(organized->begin(), organized->end(), 
        [](const PointXYZIR& a, const PointXYZIR& b) {
            if (a.ring != b.ring) return a.ring < b.ring;
            
            // Calculate azimuth for sorting within ring
            float azimuth_a = std::atan2(a.y, a.x);
            float azimuth_b = std::atan2(b.y, b.x);
            return azimuth_a < azimuth_b;
        });
    
    return organized;
}

void PointCloudProcessor::calculatePointMetrics(PointXYZIR& point) {
    // Calculate range
    point.range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    
    // Estimate ring based on vertical angle (VLP-16 specific)
    float vertical_angle = std::atan2(point.z, std::sqrt(point.x * point.x + point.y * point.y));
    
    // VLP-16 vertical angles (degrees): -15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15
    const std::vector<float> vlp16_angles = {
        -15.0f, -13.0f, -11.0f, -9.0f, -7.0f, -5.0f, -3.0f, -1.0f,
        1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f
    };
    
    float vertical_angle_deg = vertical_angle * 180.0f / M_PI;
    
    // Find closest ring
    int closest_ring = 0;
    float min_diff = std::abs(vertical_angle_deg - vlp16_angles[0]);
    
    for (size_t i = 1; i < vlp16_angles.size(); ++i) {
        float diff = std::abs(vertical_angle_deg - vlp16_angles[i]);
        if (diff < min_diff) {
            min_diff = diff;
            closest_ring = static_cast<int>(i);
        }
    }
    
    point.ring = static_cast<uint16_t>(closest_ring);
}

PointCloudXYZIR::Ptr PointCloudProcessor::transformPointCloud(
    const PointCloudXYZI::ConstPtr& input,
    const Eigen::Matrix4f& transform,
    uint8_t sensor_id) {
    
    auto output = std::make_shared<PointCloudXYZIR>();
    output->reserve(input->size());
    
    for (const auto& point_in : input->points) {
        PointXYZIR point_out;
        
        // Apply transformation
        Eigen::Vector4f point_vec(point_in.x, point_in.y, point_in.z, 1.0f);
        Eigen::Vector4f transformed = transform * point_vec;
        
        point_out.x = transformed[0];
        point_out.y = transformed[1];
        point_out.z = transformed[2];
        point_out.intensity = point_in.intensity;
        point_out.sensor_id = sensor_id;
        point_out.timestamp = 0.0f; // Will be set by timestamp synchronization
        
        // Calculate additional metrics
        calculatePointMetrics(point_out);
        
        output->push_back(point_out);
    }
    
    // Set header information
    output->header = input->header;
    output->is_dense = input->is_dense;
    
    return output;
}

} // namespace lidar_processing