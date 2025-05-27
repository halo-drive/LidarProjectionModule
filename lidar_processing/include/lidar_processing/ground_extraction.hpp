#ifndef LIDAR_PROCESSING_GROUND_EXTRACTION_HPP
#define LIDAR_PROCESSING_GROUND_EXTRACTION_HPP

#include "point_types.hpp"
#include <ros/ros.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <Eigen/Dense>

namespace lidar_processing {

class GroundExtractor {
public:
    struct GroundExtractionParams {
        // RANSAC parameters for plane fitting
        double distance_threshold;      // meters - max distance to plane
        int max_iterations;             // RANSAC iterations
        double probability;             // desired probability of success
        
        // Ring-based extraction parameters
        int start_ring;                    // lowest ring to consider
        int end_ring;                     // highest ring to consider
        double ring_height_threshold;    // height variation threshold per ring
        
        // Normal estimation parameters
        double normal_radius;            // radius for normal estimation
        int normal_k;                     // k-nearest neighbors for normals
        
        // Ground validation parameters
        double max_slope;                // maximum allowed ground slope (rad)
        double min_ground_height;       // minimum ground height (m)
        double max_ground_height;        // maximum ground height (m)
        
        // Segmentation parameters
        double segment_distance;         // distance between segments (m)
        int min_segment_points;          // minimum points per segment
        
        // Post-processing parameters
        bool use_morphological_filter;
        int morphological_iterations;
        double connectivity_radius;      // radius for connectivity check
        
        // Performance parameters
        bool use_cuda;                  // enable CUDA acceleration
        int num_threads;                   // CPU threads for processing
        
        // Default constructor with default values
        GroundExtractionParams() :
            distance_threshold(0.05),
            max_iterations(1000),
            probability(0.99),
            start_ring(0),
            end_ring(15),
            ring_height_threshold(0.2),
            normal_radius(0.5),
            normal_k(10),
            max_slope(0.3),
            min_ground_height(-2.5),
            max_ground_height(0.5),
            segment_distance(2.0),
            min_segment_points(100),
            use_morphological_filter(true),
            morphological_iterations(2),
            connectivity_radius(0.3),
            use_cuda(true),
            num_threads(4) {}
    };

    explicit GroundExtractor(const GroundExtractionParams& params = GroundExtractionParams());
    
    /**
     * @brief Extract ground plane from merged point cloud
     * @param input Input point cloud (merged from both sensors)
     * @param ground_cloud Output ground points
     * @param non_ground_cloud Output non-ground points
     * @return Ground plane coefficients [a, b, c, d] for ax + by + cz + d = 0
     */
    Eigen::Vector4f extractGroundPlane(
        const PointCloudXYZIR::ConstPtr& input,
        PointCloudGround::Ptr& ground_cloud,
        PointCloudXYZIR::Ptr& non_ground_cloud
    );
    
    /**
     * @brief Extract ground using ring-based segmentation (VLP-16 specific)
     * @param input Input organized point cloud
     * @param ground_cloud Output ground points
     * @param non_ground_cloud Output non-ground points
     * @return Success status
     */
    bool extractGroundByRings(
        const PointCloudXYZIR::ConstPtr& input,
        PointCloudGround::Ptr& ground_cloud,
        PointCloudXYZIR::Ptr& non_ground_cloud
    );
    
    /**
     * @brief Extract ground using RANSAC plane fitting
     * @param input Input point cloud
     * @param ground_cloud Output ground points
     * @param non_ground_cloud Output non-ground points
     * @return Ground plane coefficients
     */
    Eigen::Vector4f extractGroundRANSAC(
        const PointCloudXYZIR::ConstPtr& input,
        PointCloudGround::Ptr& ground_cloud,
        PointCloudXYZIR::Ptr& non_ground_cloud
    );
    
    /**
     * @brief Segment point cloud into radial sections for processing
     * @param input Input point cloud
     * @return Vector of segmented point clouds
     */
    std::vector<PointCloudXYZIR::Ptr> segmentRadially(
        const PointCloudXYZIR::ConstPtr& input
    );
    
    /**
     * @brief Estimate surface normals for ground classification
     * @param input Input point cloud
     * @return Point cloud with normal information
     */
    pcl::PointCloud<pcl::Normal>::Ptr estimateNormals(
        const PointCloudXYZIR::ConstPtr& input
    );
    
    /**
     * @brief Validate ground plane using geometric constraints
     * @param plane_coeffs Plane coefficients to validate
     * @param ground_points Points classified as ground
     * @return Validation confidence score [0.0, 1.0]
     */
    double validateGroundPlane(
        const Eigen::Vector4f& plane_coeffs,
        const PointCloudXYZIR::ConstPtr& ground_points
    );
    
    /**
     * @brief Apply morphological filtering to ground mask
     * @param ground_indices Input ground point indices
     * @param input_cloud Original point cloud
     * @return Filtered ground indices
     */
    pcl::PointIndices::Ptr morphologicalFilter(
        const pcl::PointIndices::ConstPtr& ground_indices,
        const PointCloudXYZIR::ConstPtr& input_cloud
    );
    
    /**
     * @brief Convert point cloud to ground point type with confidence
     * @param input Input point cloud
     * @param ground_confidence Confidence values for each point
     * @param normals Surface normals
     * @return Ground point cloud
     */
    PointCloudGround::Ptr convertToGroundPoints(
        const PointCloudXYZIR::ConstPtr& input,
        const std::vector<float>& ground_confidence,
        const pcl::PointCloud<pcl::Normal>::ConstPtr& normals
    );
    
    // Configuration methods
    void setParams(const GroundExtractionParams& params) { params_ = params; }
    const GroundExtractionParams& getParams() const { return params_; }
    
    // Performance monitoring
    struct ExtractionStats {
        size_t input_points;
        size_t ground_points;
        size_t non_ground_points;
        double extraction_time_ms;
        double normal_estimation_time_ms;
        double validation_confidence;
        Eigen::Vector4f ground_plane_coeffs;
        
        ExtractionStats() : 
            input_points(0), ground_points(0), non_ground_points(0),
            extraction_time_ms(0.0), normal_estimation_time_ms(0.0),
            validation_confidence(0.0), ground_plane_coeffs(Eigen::Vector4f::Zero()) {}
    };
    
    const ExtractionStats& getLastStats() const { return last_stats_; }
    
    /**
     * @brief Get ground height at specific x,y coordinate
     * @param x X coordinate
     * @param y Y coordinate
     * @return Ground height (z coordinate)
     */
    float getGroundHeightAt(float x, float y) const;
    
    /**
     * @brief Check if point is likely to be on ground
     * @param point Point to check
     * @return Ground probability [0.0, 1.0]
     */
    float classifyPoint(const PointXYZIR& point) const;

private:
    GroundExtractionParams params_;
    ExtractionStats last_stats_;
    
    // Cached ground plane for height queries
    Eigen::Vector4f current_ground_plane_;
    bool ground_plane_valid_;
    
    // PCL objects (reused for efficiency)
    pcl::SACSegmentation<PointXYZIR> ransac_seg_;
    pcl::ExtractIndices<PointXYZIR> extract_indices_;
    pcl::NormalEstimation<PointXYZIR, pcl::Normal> normal_estimator_;
    pcl::search::KdTree<PointXYZIR>::Ptr kdtree_;
    
    /**
     * @brief Process individual ring for ground extraction
     * @param ring_points Points from a specific ring
     * @param ring_id Ring identifier
     * @return Ground indices for this ring
     */
    pcl::PointIndices::Ptr processRing(
        const PointCloudXYZIR::ConstPtr& ring_points,
        int ring_id
    );
    
    /**
     * @brief Calculate ground confidence based on multiple criteria
     * @param point Point to evaluate
     * @param normal Surface normal at point
     * @param plane_distance Distance to fitted plane
     * @return Confidence score [0.0, 1.0]
     */
    float calculateGroundConfidence(
        const PointXYZIR& point,
        const pcl::Normal& normal,
        float plane_distance
    ) const;
    
    /**
     * @brief Merge ground indices from different processing methods
     * @param ransac_indices Indices from RANSAC
     * @param ring_indices Indices from ring-based method
     * @param input_cloud Original point cloud
     * @return Merged and validated ground indices
     */
    pcl::PointIndices::Ptr mergeGroundIndices(
        const pcl::PointIndices::ConstPtr& ransac_indices,
        const pcl::PointIndices::ConstPtr& ring_indices,
        const PointCloudXYZIR::ConstPtr& input_cloud
    );
};

} // namespace lidar_processing

#endif // LIDAR_PROCESSING_GROUND_EXTRACTION_HPP