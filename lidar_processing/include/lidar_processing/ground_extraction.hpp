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
#include <pcl/common/common.h>
#include <Eigen/Dense>

namespace lidar_processing {

class GroundExtractor {
public:
    struct GroundExtractionParams {
        // IMPROVED: Grid-based preprocessing parameters (Livox-inspired)
        float grid_size_x;              // Grid cell size in X direction (4m like Livox)
        float grid_size_y;              // Grid cell size in Y direction (4m like Livox) 
        float height_tolerance;         // Height variation tolerance within grid (0.4m like Livox)
        
        // IMPROVED: Much stricter validation parameters
        float max_ground_height;        // Absolute maximum ground height
        float min_ground_height;        // Absolute minimum ground height
        float normal_z_threshold;       // CRITICAL: Minimum Z component of normal (0.85 - much stricter!)
        
        // RANSAC parameters - MUCH TIGHTER than your current ones
        double distance_threshold;      // CHANGED: 0.02m instead of your 0.05m
        int max_iterations;             // Reduced iterations since we pre-filter
        double probability;             // Keep same
        
        // Performance parameters
        bool use_cuda;
        int num_threads;
        
        // UPDATED DEFAULT VALUES - Key changes marked with comments
        GroundExtractionParams() :
            // NEW: Grid-based preprocessing
            grid_size_x(4.0f),          // NEW: 4m grid like Livox
            grid_size_y(4.0f),          // NEW: 4m grid like Livox  
            height_tolerance(0.4f),     // NEW: 40cm tolerance like Livox
            
            // STRICTER validation
            max_ground_height(0.5f),    // SAME but will be enforced better
            min_ground_height(-2.5f),   // SAME but will be enforced better
            normal_z_threshold(0.85f),  // NEW: CRITICAL - normal must point upward!
            
            // MUCH TIGHTER RANSAC
            distance_threshold(0.02),   // CHANGED: 0.02m instead of your 0.05m
            max_iterations(500),        // REDUCED: 500 instead of your 1000
            probability(0.99),          // SAME
            
            use_cuda(true),
            num_threads(4) {}
    };

    explicit GroundExtractor(const GroundExtractionParams& params = GroundExtractionParams());
    
    // SIMPLIFIED: Main ground extraction method using standard PCL types
    Eigen::Vector4f extractGroundPlane(
        const PointCloudXYZI::ConstPtr& input,      // CHANGED: Use standard PCL type
        PointCloudXYZI::Ptr& ground_cloud,          // CHANGED: Use standard PCL type  
        PointCloudXYZI::Ptr& non_ground_cloud       // CHANGED: Use standard PCL type
    );
    
    // Configuration methods (keep same)
    void setParams(const GroundExtractionParams& params) { params_ = params; }
    const GroundExtractionParams& getParams() const { return params_; }
    
    // Performance monitoring (simplified)
    struct ExtractionStats {
        size_t input_points;
        size_t ground_points;
        size_t non_ground_points;
        double extraction_time_ms;
        double validation_confidence;
        Eigen::Vector4f ground_plane_coeffs;
        
        ExtractionStats() : 
            input_points(0), ground_points(0), non_ground_points(0),
            extraction_time_ms(0.0), validation_confidence(0.0), 
            ground_plane_coeffs(Eigen::Vector4f::Zero()) {}
    };
    
    const ExtractionStats& getLastStats() const { return last_stats_; }

private:
    GroundExtractionParams params_;
    ExtractionStats last_stats_;
    
    // Cached ground plane for height queries
    Eigen::Vector4f current_ground_plane_;
    bool ground_plane_valid_;
    
    // PCL objects (reused for efficiency) - SIMPLIFIED
    pcl::SACSegmentation<pcl::PointXYZI> ransac_seg_;           // CHANGED: Use standard PCL type
    pcl::ExtractIndices<pcl::PointXYZI> extract_indices_;       // CHANGED: Use standard PCL type
    
    // SIMPLIFIED: Grid-based preprocessing methods (Livox-inspired) - Using standard types
    std::vector<int> gridBasedPreprocessing(const PointCloudXYZI::ConstPtr& input);
    Eigen::Vector4f pcaPlaneEstimation(const PointCloudXYZI::ConstPtr& input, 
                                       const std::vector<int>& candidate_indices);
    
    // SIMPLIFIED: Validation method using standard types
    double validateGroundPlane(
        const Eigen::Vector4f& plane_coeffs,
        const PointCloudXYZI::ConstPtr& ground_points
    );
};

} // namespace lidar_processing

#endif // LIDAR_PROCESSING_GROUND_EXTRACTION_HPP


// #ifndef LIDAR_PROCESSING_GROUND_EXTRACTION_HPP
// #define LIDAR_PROCESSING_GROUND_EXTRACTION_HPP

// #include "point_types.hpp"
// #include <ros/ros.h>
// #include <pcl/segmentation/sac_segmentation.h>
// #include <pcl/filters/extract_indices.h>
// #include <pcl/sample_consensus/method_types.h>
// #include <pcl/sample_consensus/model_types.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/kdtree/kdtree.h>
// #include <pcl/common/common.h>
// #include <Eigen/Dense>

// namespace lidar_processing {

// class GroundExtractor {
// public:
//     struct GroundExtractionParams {
//         // IMPROVED: Grid-based preprocessing parameters (Livox-inspired)
//         float grid_size_x;              // Grid cell size in X direction (4m like Livox)
//         float grid_size_y;              // Grid cell size in Y direction (4m like Livox) 
//         float height_tolerance;         // Height variation tolerance within grid (0.4m like Livox)
        
//         // IMPROVED: Much stricter validation parameters
//         float max_ground_height;        // Absolute maximum ground height (0.5m instead of your 0.5m)
//         float min_ground_height;        // Absolute minimum ground height (-2.5m)
//         float normal_z_threshold;       // CRITICAL: Minimum Z component of normal (0.85 - much stricter!)
        
//         // RANSAC parameters - MUCH TIGHTER than your current ones
//         double distance_threshold;      // CHANGED: 0.02m instead of your 0.05m
//         int max_iterations;             // Reduced iterations since we pre-filter
//         double probability;             // Keep same
        
//         // Ring-based extraction parameters (keep your existing)
//         int start_ring;
//         int end_ring;
//         double ring_height_threshold;
        
//         // Normal estimation parameters (keep your existing)
//         double normal_radius;
//         int normal_k;
        
//         // Ground validation parameters - MUCH STRICTER
//         double max_slope;               // Keep same
//         double segment_distance;        // Keep same  
//         int min_segment_points;         // Keep same
        
//         // Post-processing parameters (keep your existing)
//         bool use_morphological_filter;
//         int morphological_iterations;
//         double connectivity_radius;
        
//         // Performance parameters
//         bool use_cuda;
//         int num_threads;
        
//         // UPDATED DEFAULT VALUES - Key changes marked with comments
//         GroundExtractionParams() :
//             // NEW: Grid-based preprocessing
//             grid_size_x(4.0f),          // NEW: 4m grid like Livox
//             grid_size_y(4.0f),          // NEW: 4m grid like Livox  
//             height_tolerance(0.4f),     // NEW: 40cm tolerance like Livox
            
//             // STRICTER validation
//             max_ground_height(0.5f),    // SAME but will be enforced better
//             min_ground_height(-2.5f),   // SAME but will be enforced better
//             normal_z_threshold(0.85f),  // NEW: CRITICAL - normal must point upward!
            
//             // MUCH TIGHTER RANSAC
//             distance_threshold(0.02),   // CHANGED: 0.02m instead of your 0.05m
//             max_iterations(500),        // REDUCED: 500 instead of your 1000
//             probability(0.99),          // SAME
            
//             // Keep your existing values
//             start_ring(0),
//             end_ring(15), 
//             ring_height_threshold(0.2),
//             normal_radius(0.5),
//             normal_k(10),
//             max_slope(0.3),
//             segment_distance(2.0),
//             min_segment_points(100),
//             use_morphological_filter(true),
//             morphological_iterations(2),
//             connectivity_radius(0.3),
//             use_cuda(true),
//             num_threads(4) {}
//     };

//     explicit GroundExtractor(const GroundExtractionParams& params = GroundExtractionParams());
    
//     // Keep all your existing public methods - no changes needed
//     Eigen::Vector4f extractGroundPlane(
//         const PointCloudXYZIR::ConstPtr& input,
//         PointCloudGround::Ptr& ground_cloud,
//         PointCloudXYZIR::Ptr& non_ground_cloud
//     );
    
//     bool extractGroundByRings(
//         const PointCloudXYZIR::ConstPtr& input,
//         PointCloudGround::Ptr& ground_cloud,
//         PointCloudXYZIR::Ptr& non_ground_cloud
//     );
    
//     Eigen::Vector4f extractGroundRANSAC(
//         const PointCloudXYZIR::ConstPtr& input,
//         PointCloudGround::Ptr& ground_cloud,
//         PointCloudXYZIR::Ptr& non_ground_cloud
//     );
    
//     std::vector<PointCloudXYZIR::Ptr> segmentRadially(
//         const PointCloudXYZIR::ConstPtr& input
//     );
    
//     pcl::PointCloud<pcl::Normal>::Ptr estimateNormals(
//         const PointCloudXYZIR::ConstPtr& input
//     );
    
//     double validateGroundPlane(
//         const Eigen::Vector4f& plane_coeffs,
//         const PointCloudXYZIR::ConstPtr& ground_points
//     );
    
//     pcl::PointIndices::Ptr morphologicalFilter(
//         const pcl::PointIndices::ConstPtr& ground_indices,
//         const PointCloudXYZIR::ConstPtr& input_cloud
//     );
    
//     PointCloudGround::Ptr convertToGroundPoints(
//         const PointCloudXYZIR::ConstPtr& input,
//         const std::vector<float>& ground_confidence,
//         const pcl::PointCloud<pcl::Normal>::ConstPtr& normals
//     );
    
//     // Configuration methods (keep same)
//     void setParams(const GroundExtractionParams& params) { params_ = params; }
//     const GroundExtractionParams& getParams() const { return params_; }
    
//     // Performance monitoring (keep same)
//     struct ExtractionStats {
//         size_t input_points;
//         size_t ground_points;
//         size_t non_ground_points;
//         double extraction_time_ms;
//         double normal_estimation_time_ms;
//         double validation_confidence;
//         Eigen::Vector4f ground_plane_coeffs;
        
//         ExtractionStats() : 
//             input_points(0), ground_points(0), non_ground_points(0),
//             extraction_time_ms(0.0), normal_estimation_time_ms(0.0),
//             validation_confidence(0.0), ground_plane_coeffs(Eigen::Vector4f::Zero()) {}
//     };
    
//     const ExtractionStats& getLastStats() const { return last_stats_; }
    
//     float getGroundHeightAt(float x, float y) const;
//     float classifyPoint(const PointXYZIR& point) const;

// private:
//     GroundExtractionParams params_;
//     ExtractionStats last_stats_;
    
//     // Cached ground plane for height queries
//     Eigen::Vector4f current_ground_plane_;
//     bool ground_plane_valid_;
    
//     // PCL objects (reused for efficiency)
//     pcl::SACSegmentation<PointXYZIR> ransac_seg_;
//     pcl::ExtractIndices<PointXYZIR> extract_indices_;
//     pcl::NormalEstimation<PointXYZIR, pcl::Normal> normal_estimator_;
//     pcl::search::KdTree<PointXYZIR>::Ptr kdtree_;
    
//     // NEW: Grid-based preprocessing methods (Livox-inspired)
//     std::vector<int> gridBasedPreprocessing(const PointCloudXYZIR::ConstPtr& input);
//     Eigen::Vector4f pcaPlaneEstimation(const PointCloudXYZIR::ConstPtr& input, 
//                                        const std::vector<int>& candidate_indices);
    
//     // Keep your existing private methods
//     pcl::PointIndices::Ptr processRing(
//         const PointCloudXYZIR::ConstPtr& ring_points,
//         int ring_id
//     );
    
//     float calculateGroundConfidence(
//         const PointXYZIR& point,
//         const pcl::Normal& normal,
//         float plane_distance
//     ) const;
    
//     pcl::PointIndices::Ptr mergeGroundIndices(
//         const pcl::PointIndices::ConstPtr& ransac_indices,
//         const pcl::PointIndices::ConstPtr& ring_indices,
//         const PointCloudXYZIR::ConstPtr& input_cloud
//     );
// };

// } // namespace lidar_processing

// #endif // LIDAR_PROCESSING_GROUND_EXTRACTION_HPP
