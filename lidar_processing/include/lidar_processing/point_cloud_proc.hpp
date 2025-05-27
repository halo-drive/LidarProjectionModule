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
        int statistical_k;             // k-nearest neighbors
        double statistical_std_mul;   // standard deviation multiplier
        
        // Radius outlier removal
        double radius_search;         // search radius in meters
        int min_neighbors;              // minimum neighbors in radius
        
        // Range filtering
        float min_range;
        float max_range;
        
        // Z-axis filtering (height)
        float min_height;           // meters below sensor
        float max_height;            // meters above sensor
        
        // Default constructor
        FilterParams() :
            voxel_size(0.1f),
            statistical_k(50),
            statistical_std_mul(1.0),
            radius_search(0.5),
            min_neighbors(5),
            min_range(VLP16_MIN_RANGE),
            max_range(VLP16_MAX_RANGE),
            min_height(-3.0f),
            max_height(5.0f) {}
    };

    explicit PointCloudProcessor(const FilterParams& params = FilterParams());
    
    /**
     * @brief Merge point clouds from two VLP-16 sensors
     * @param cloud1 Point cloud from first sensor (puck)
     * @param cloud2 Point cloud from second sensor (high-res puck)
     * @param transform2 Transformation matrix for second sensor relative to first
     * @return Merged point cloud with sensor IDs
     */
    PointCloudXYZIR::Ptr mergePointClouds(
        const PointCloudXYZI::ConstPtr& cloud1,
        const PointCloudXYZI::ConstPtr& cloud2,
        const Eigen::Matrix4f& transform2
    );
    
    /**
     * @brief Convert ROS PointCloud2 message to custom point type
     * @param msg ROS PointCloud2 message
     * @param sensor_id Sensor identifier
     * @return Converted point cloud
     */
    PointCloudXYZIR::Ptr convertFromROS(
        const sensor_msgs::PointCloud2::ConstPtr& msg,
        std::uint8_t sensor_id
    );
    
    /**
     * @brief Apply filtering pipeline to point cloud
     * @param input Input point cloud
     * @return Filtered point cloud
     */
    PointCloudXYZIR::Ptr applyFilters(const PointCloudXYZIR::ConstPtr& input);
    
    /**
     * @brief Remove points based on range and height constraints
     * @param input Input point cloud
     * @return Range-filtered point cloud
     */
    PointCloudXYZIR::Ptr rangeFilter(const PointCloudXYZIR::ConstPtr& input);
    
    /**
     * @brief Downsample point cloud using voxel grid
     * @param input Input point cloud
     * @return Downsampled point cloud
     */
    PointCloudXYZIR::Ptr voxelGridFilter(const PointCloudXYZIR::ConstPtr& input);
    
    /**
     * @brief Remove statistical outliers
     * @param input Input point cloud
     * @return Outlier-filtered point cloud
     */
    PointCloudXYZIR::Ptr statisticalOutlierFilter(const PointCloudXYZIR::ConstPtr& input);
    
    /**
     * @brief Remove points with few neighbors in radius
     * @param input Input point cloud
     * @return Radius-filtered point cloud
     */
    PointCloudXYZIR::Ptr radiusOutlierFilter(const PointCloudXYZIR::ConstPtr& input);
    
    /**
     * @brief Organize point cloud by ring structure for VLP-16
     * @param input Input point cloud
     * @return Ring-organized point cloud
     */
    PointCloudXYZIR::Ptr organizeByRings(const PointCloudXYZIR::ConstPtr& input);
    
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
        
        ProcessingStats() : 
            input_points(0), output_points(0), processing_time_ms(0.0), 
            merge_time_ms(0.0), filter_time_ms(0.0) {}
    };
    
    const ProcessingStats& getLastStats() const { return last_stats_; }

private:
    FilterParams params_;
    ProcessingStats last_stats_;
    
    // PCL filter objects (reused for efficiency)
    pcl::VoxelGrid<PointXYZIR> voxel_filter_;
    pcl::StatisticalOutlierRemoval<PointXYZIR> statistical_filter_;
    pcl::RadiusOutlierRemoval<PointXYZIR> radius_filter_;
    
    /**
     * @brief Calculate range and ring information for VLP-16 point
     * @param point Input point to analyze
     */
    void calculatePointMetrics(PointXYZIR& point);
    
    /**
     * @brief Transform point cloud using transformation matrix
     * @param input Input point cloud
     * @param transform Transformation matrix
     * @return Transformed point cloud
     */
    PointCloudXYZIR::Ptr transformPointCloud(
        const PointCloudXYZI::ConstPtr& input,
        const Eigen::Matrix4f& transform,
        std::uint8_t sensor_id
    );
};

} // namespace lidar_processing

#endif // LIDAR_PROCESSING_POINT_CLOUD_PROC_HPP