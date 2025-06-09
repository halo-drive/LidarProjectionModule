#ifndef LIDAR_PROCESSING_POINT_CLOUD_PROC_HPP
#define LIDAR_PROCESSING_POINT_CLOUD_PROC_HPP

#include "point_types.hpp"
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <memory>

namespace lidar_processing {

class PointCloudProcessor {
public:
    struct FilterParams {
        // Voxel grid parameters
        float voxel_size;            // meters
        
        // Statistical outlier removal
        int statistical_k;           // k-nearest neighbors
        double statistical_std_mul;  // standard deviation multiplier
        
        // Radius outlier removal
        double radius_search;        // search radius in meters
        int min_neighbors;           // minimum neighbors in radius
        
        // Range filtering
        float min_range;
        float max_range;
        
        // Z-axis filtering (height)
        float min_height;           // meters below sensor
        float max_height;           // meters above sensor
        
        // Default constructor with SAFE defaults
        FilterParams() :
            voxel_size(0.1f),
            statistical_k(50),
            statistical_std_mul(1.0),
            radius_search(0.5),
            min_neighbors(5),
            min_range(0.9f),        // FIXED: Use actual values instead of undefined constants
            max_range(100.0f),      // FIXED: Use actual values instead of undefined constants
            min_height(-3.0f),
            max_height(5.0f) {}
    };

    explicit PointCloudProcessor(const FilterParams& params = FilterParams());
    
    /**
     * @brief Merge point clouds from two VLP-16 sensors
     * SIMPLIFIED: Use standard PCL types only
     */
    PointCloudXYZI::Ptr mergePointClouds(
        const PointCloudXYZI::ConstPtr& cloud1,
        const PointCloudXYZI::ConstPtr& cloud2,
        const Eigen::Matrix4f& transform2
    );
    
    /**
     * @brief Convert ROS PointCloud2 message to standard PCL type
     * SIMPLIFIED: No custom sensor_id needed
     */
    PointCloudXYZI::Ptr convertFromROS(
        const sensor_msgs::PointCloud2::ConstPtr& msg
    );
    
    /**
     * @brief Apply filtering pipeline to point cloud
     * SIMPLIFIED: Use standard PCL types
     */
    PointCloudXYZI::Ptr applyFilters(const PointCloudXYZI::ConstPtr& input);
    
    /**
     * @brief Remove points based on range and height constraints
     */
    PointCloudXYZI::Ptr rangeFilter(const PointCloudXYZI::ConstPtr& input);
    
    /**
     * @brief Downsample point cloud using voxel grid
     */
    PointCloudXYZI::Ptr voxelGridFilter(const PointCloudXYZI::ConstPtr& input);
    
    /**
     * @brief Remove statistical outliers
     */
    PointCloudXYZI::Ptr statisticalOutlierFilter(const PointCloudXYZI::ConstPtr& input);
    
    /**
     * @brief Remove points with few neighbors in radius
     */
    PointCloudXYZI::Ptr radiusOutlierFilter(const PointCloudXYZI::ConstPtr& input);
    
    // Getters and setters
    void setFilterParams(const FilterParams& params) { params_ = params; }
    const FilterParams& getFilterParams() const { return params_; }
    
    // Performance monitoring
    struct ProcessingStats {
        size_t input_points;
        size_t output_points;
        double processing_time_ms;
        double merge_time_ms;
        double filter_time_ms;
        
        // FIXED: Proper constructor syntax
        ProcessingStats() : 
            input_points(0), output_points(0), processing_time_ms(0.0), 
            merge_time_ms(0.0), filter_time_ms(0.0) {}
    };
    
    const ProcessingStats& getLastStats() const { return last_stats_; }

private:
    FilterParams params_;
    ProcessingStats last_stats_;
    
    // PCL filter objects (reused for efficiency) - USING STANDARD PCL TYPES
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter_;
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> statistical_filter_;
    pcl::RadiusOutlierRemoval<pcl::PointXYZI> radius_filter_;
    
    /**
     * @brief Transform point cloud using transformation matrix
     * SIMPLIFIED: Standard PCL types only
     */
    PointCloudXYZI::Ptr transformPointCloud(
        const PointCloudXYZI::ConstPtr& input,
        const Eigen::Matrix4f& transform
    );
};

} // namespace lidar_processing

#endif // LIDAR_PROCESSING_POINT_CLOUD_PROC_HPP
