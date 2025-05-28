#include "lidar_processing/point_cloud_proc.hpp"
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <chrono>
#include <cmath>

namespace lidar_processing {

PointCloudProcessor::PointCloudProcessor(const FilterParams& params)
    : params_(params) {
    
    // Configure PCL filters - USING STANDARD PCL TYPES
    voxel_filter_.setLeafSize(params_.voxel_size, params_.voxel_size, params_.voxel_size);
    
    statistical_filter_.setMeanK(params_.statistical_k);
    statistical_filter_.setStddevMulThresh(params_.statistical_std_mul);
    
    radius_filter_.setRadiusSearch(params_.radius_search);
    radius_filter_.setMinNeighborsInRadius(params_.min_neighbors);
}

// CORRECTED: Method signature matches header
PointCloudXYZI::Ptr PointCloudProcessor::mergePointClouds(
    const PointCloudXYZI::ConstPtr& cloud1,
    const PointCloudXYZI::ConstPtr& cloud2,
    const Eigen::Matrix4f& transform2) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Reset statistics
    last_stats_ = ProcessingStats();
    last_stats_.input_points = cloud1->size() + cloud2->size();
    
    auto merged_cloud = boost::make_shared<PointCloudXYZI>(); // CORRECTED: Use standard type
    
    // Reserve memory for efficiency
    merged_cloud->reserve(cloud1->size() + cloud2->size());
    
    // Process first sensor (no transformation needed)
    auto cloud1_converted = transformPointCloud(cloud1, Eigen::Matrix4f::Identity());
    
    // Process second sensor (apply transformation)
    auto cloud2_converted = transformPointCloud(cloud2, transform2);
    
    // Merge the clouds - SIMPLIFIED
    *merged_cloud = *cloud1_converted;
    *merged_cloud += *cloud2_converted;
    
    // Set header information
    merged_cloud->header = cloud1->header;
    merged_cloud->is_dense = false; // May contain NaN/Inf values
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.merge_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    last_stats_.output_points = merged_cloud->size();
    
    ROS_INFO("Merged %zu + %zu points into %zu points in %.2f ms",
        cloud1->size(), cloud2->size(), merged_cloud->size(), last_stats_.merge_time_ms);
    
    return merged_cloud;
}

// CORRECTED: Method signature matches header (no sensor_id parameter)
PointCloudXYZI::Ptr PointCloudProcessor::convertFromROS(
    const sensor_msgs::PointCloud2::ConstPtr& msg) {
    
    // Convert ROS message to PCL - SIMPLIFIED
    auto pcl_cloud = boost::make_shared<PointCloudXYZI>();
    pcl::fromROSMsg(*msg, *pcl_cloud);
    
    return pcl_cloud; // SIMPLIFIED: No transformation needed
}

// CORRECTED: Method signature matches header
PointCloudXYZI::Ptr PointCloudProcessor::applyFilters(const PointCloudXYZI::ConstPtr& input) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ROS_DEBUG("Applying filters to %zu points", input->size());
    
    // Apply filters in sequence - SIMPLIFIED
    auto filtered = rangeFilter(input);
    filtered = voxelGridFilter(filtered);
    filtered = statisticalOutlierFilter(filtered);
    filtered = radiusOutlierFilter(filtered);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.filter_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    ROS_INFO("Filtered %zu -> %zu points in %.2f ms",
        input->size(), filtered->size(), last_stats_.filter_time_ms);
    
    return filtered;
}

// CORRECTED: Method signature matches header
PointCloudXYZI::Ptr PointCloudProcessor::rangeFilter(const PointCloudXYZI::ConstPtr& input) {
    auto filtered = boost::make_shared<PointCloudXYZI>();
    filtered->reserve(input->size());
    
    for (const auto& point : input->points) {
        // Skip invalid points
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            continue;
        }
        
        // Compute range
        float range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        
        // Check range constraints
        if (range >= params_.min_range && range <= params_.max_range &&
            point.z >= params_.min_height && point.z <= params_.max_height) {
            filtered->push_back(point);
        }
    }
    
    filtered->header = input->header;
    filtered->is_dense = true;
    
    return filtered;
}

// CORRECTED: Method signature matches header
PointCloudXYZI::Ptr PointCloudProcessor::voxelGridFilter(const PointCloudXYZI::ConstPtr& input) {
    auto filtered = boost::make_shared<PointCloudXYZI>();
    
    voxel_filter_.setInputCloud(input);
    voxel_filter_.filter(*filtered);
    
    return filtered;
}

// CORRECTED: Method signature matches header
PointCloudXYZI::Ptr PointCloudProcessor::statisticalOutlierFilter(const PointCloudXYZI::ConstPtr& input) {
    auto filtered = boost::make_shared<PointCloudXYZI>();
    
    statistical_filter_.setInputCloud(input);
    statistical_filter_.filter(*filtered);
    
    return filtered;
}

// CORRECTED: Method signature matches header
PointCloudXYZI::Ptr PointCloudProcessor::radiusOutlierFilter(const PointCloudXYZI::ConstPtr& input) {
    auto filtered = boost::make_shared<PointCloudXYZI>();
    
    radius_filter_.setInputCloud(input);
    radius_filter_.filter(*filtered);
    
    return filtered;
}

// CORRECTED: Method signature matches header (no sensor_id parameter)
PointCloudXYZI::Ptr PointCloudProcessor::transformPointCloud(
    const PointCloudXYZI::ConstPtr& input,
    const Eigen::Matrix4f& transform) {
    
    auto output = boost::make_shared<PointCloudXYZI>();
    output->reserve(input->size());
    
    for (const auto& point_in : input->points) {
        pcl::PointXYZI point_out;
        
        // Apply transformation
        Eigen::Vector4f point_vec(point_in.x, point_in.y, point_in.z, 1.0f);
        Eigen::Vector4f transformed = transform * point_vec;
        
        point_out.x = transformed[0];
        point_out.y = transformed[1];
        point_out.z = transformed[2];
        point_out.intensity = point_in.intensity;
        
        output->push_back(point_out);
    }
    
    // Set header information
    output->header = input->header;
    output->is_dense = input->is_dense;
    
    return output;
}

} // namespace lidar_processing
