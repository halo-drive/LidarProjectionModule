#include "lidar_processing/ground_extraction.hpp"
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <algorithm>
#include <chrono>

namespace lidar_processing {

GroundExtractor::GroundExtractor(const GroundExtractionParams& params)
    : params_(params), ground_plane_valid_(false) {
    
    // Configure RANSAC segmentation - MUCH STRICTER
    ransac_seg_.setOptimizeCoefficients(true);
    ransac_seg_.setModelType(pcl::SACMODEL_PLANE);
    ransac_seg_.setMethodType(pcl::SAC_RANSAC);
    ransac_seg_.setMaxIterations(params_.max_iterations);
    ransac_seg_.setDistanceThreshold(params_.distance_threshold);
    ransac_seg_.setProbability(params_.probability);
    
    // Configure extraction
    extract_indices_.setNegative(false);
    
    ROS_INFO("GroundExtractor initialized with STRICT parameters:");
    ROS_INFO("  Distance threshold: %.3f m", params_.distance_threshold);
    ROS_INFO("  Normal Z threshold: %.2f", params_.normal_z_threshold);
    ROS_INFO("  Max height: %.1f m", params_.max_ground_height);
}

// SIMPLIFIED: Grid-based preprocessing using standard PCL types
std::vector<int> GroundExtractor::gridBasedPreprocessing(const PointCloudXYZI::ConstPtr& input) {
    std::vector<int> ground_candidates;
    
    // Define grid bounds
    float min_x = -50.0f, max_x = 50.0f;  // 100m x 100m area
    float min_y = -50.0f, max_y = 50.0f;
    
    int grid_width = (int)((max_x - min_x) / params_.grid_size_x) + 1;
    int grid_height = (int)((max_y - min_y) / params_.grid_size_y) + 1;
    
    // Grid to store minimum height in each cell
    std::vector<std::vector<float>> grid_min_height(grid_height, std::vector<float>(grid_width, FLT_MAX));
    std::vector<std::vector<int>> grid_point_count(grid_height, std::vector<int>(grid_width, 0));
    
    // First pass: Find minimum height in each grid cell
    for (size_t i = 0; i < input->size(); ++i) {
        const auto& point = input->points[i];
        
        // Skip points outside height bounds
        if (point.z > params_.max_ground_height || point.z < params_.min_ground_height) {
            continue;
        }
        
        // Calculate grid coordinates
        int grid_x = (int)((point.x - min_x) / params_.grid_size_x);
        int grid_y = (int)((point.y - min_y) / params_.grid_size_y);
        
        if (grid_x >= 0 && grid_x < grid_width && grid_y >= 0 && grid_y < grid_height) {
            if (point.z < grid_min_height[grid_y][grid_x]) {
                grid_min_height[grid_y][grid_x] = point.z;
            }
            grid_point_count[grid_y][grid_x]++;
        }
    }
    
    // Second pass: Select ground candidates based on Livox logic
    for (size_t i = 0; i < input->size(); ++i) {
        const auto& point = input->points[i];
        
        // Skip points outside height bounds
        if (point.z > params_.max_ground_height || point.z < params_.min_ground_height) {
            continue;
        }
        
        // Calculate grid coordinates
        int grid_x = (int)((point.x - min_x) / params_.grid_size_x);
        int grid_y = (int)((point.y - min_y) / params_.grid_size_y);
        
        if (grid_x >= 0 && grid_x < grid_width && grid_y >= 0 && grid_y < grid_height) {
            float min_height_in_cell = grid_min_height[grid_y][grid_x];
            int points_in_cell = grid_point_count[grid_y][grid_x];
            
            // Livox-inspired ground candidate selection:
            // 1. Point must be close to minimum height in cell
            bool close_to_min = (point.z - min_height_in_cell) <= params_.height_tolerance;
            
            // 2. Cell must have sufficient point density
            bool sufficient_density = points_in_cell >= 3;
            
            // 3. Additional height-based rules (from Livox)
            bool height_valid = true;
            if (point.z > 1.0f) {  // Too high = not ground
                height_valid = false;
            }
            
            // 4. Distance-based height validation
            float horizontal_dist = std::sqrt(point.x * point.x + point.y * point.y);
            if (horizontal_dist < 10.0f && point.z > 0.5f) {  // Close points shouldn't be too high
                height_valid = false;
            }
            
            if (close_to_min && sufficient_density && height_valid) {
                ground_candidates.push_back(i);
            }
        }
    }
    
    ROS_INFO("Grid-based preprocessing: %zu candidates from %zu points", 
             ground_candidates.size(), input->size());
    
    return ground_candidates;
}

// SIMPLIFIED: PCA-based plane estimation using standard PCL types
Eigen::Vector4f GroundExtractor::pcaPlaneEstimation(const PointCloudXYZI::ConstPtr& input, 
                                                     const std::vector<int>& candidate_indices) {
    if (candidate_indices.size() < 100) {
        ROS_WARN("Too few candidates for PCA: %zu", candidate_indices.size());
        return Eigen::Vector4f::Zero();
    }
    
    // Compute centroid
    Eigen::Vector3f centroid(0, 0, 0);
    for (int idx : candidate_indices) {
        const auto& point = input->points[idx];
        centroid += Eigen::Vector3f(point.x, point.y, point.z);
    }
    centroid /= candidate_indices.size();
    
    // Compute covariance matrix
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (int idx : candidate_indices) {
        const auto& point = input->points[idx];
        Eigen::Vector3f centered = Eigen::Vector3f(point.x, point.y, point.z) - centroid;
        covariance += centered * centered.transpose();
    }
    covariance /= (candidate_indices.size() - 1);
    
    // Compute eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance);
    Eigen::Vector3f eigenvalues = eigen_solver.eigenvalues();
    Eigen::Matrix3f eigenvectors = eigen_solver.eigenvectors();
    
    // The normal is the eigenvector corresponding to the smallest eigenvalue
    Eigen::Vector3f normal = eigenvectors.col(0);  // Smallest eigenvalue is first
    
    // Ensure normal points upward (CRITICAL for avoiding wall detection!)
    if (normal.z() < 0) {
        normal = -normal;
    }
    
    // Check if normal is sufficiently upward-pointing (CRITICAL CHECK!)
    if (normal.z() < params_.normal_z_threshold) {
        ROS_WARN("PCA plane normal not upward enough: z=%.3f (threshold=%.3f)", 
                 normal.z(), params_.normal_z_threshold);
        return Eigen::Vector4f::Zero();
    }
    
    // Compute d coefficient: normal · centroid + d = 0 => d = -normal · centroid
    float d = -normal.dot(centroid);
    
    ROS_INFO("PCA plane: normal=[%.3f, %.3f, %.3f], d=%.3f", 
             normal.x(), normal.y(), normal.z(), d);
    
    return Eigen::Vector4f(normal.x(), normal.y(), normal.z(), d);
}

// MAIN EXTRACTION METHOD - SIMPLIFIED and ROBUST
Eigen::Vector4f GroundExtractor::extractGroundPlane(
    const PointCloudXYZI::ConstPtr& input,
    PointCloudXYZI::Ptr& ground_cloud,
    PointCloudXYZI::Ptr& non_ground_cloud) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Reset statistics
    last_stats_ = ExtractionStats();
    last_stats_.input_points = input->size();
    
    ROS_INFO("Starting IMPROVED ground extraction with %zu points", input->size());
    
    // Initialize output clouds
    ground_cloud.reset(new PointCloudXYZI());
    non_ground_cloud.reset(new PointCloudXYZI());
    
    if (input->empty()) {
        ROS_WARN("Input cloud is empty");
        return Eigen::Vector4f::Zero();
    }
    
    // STEP 1: Grid-based preprocessing (NEW - Livox-inspired)
    auto ground_candidates = gridBasedPreprocessing(input);
    
    if (ground_candidates.size() < 100) {
        ROS_ERROR("Too few ground candidates after preprocessing: %zu", ground_candidates.size());
        return Eigen::Vector4f::Zero();
    }
    
    // STEP 2: PCA-based initial plane estimation (NEW)
    Eigen::Vector4f plane_coeffs = pcaPlaneEstimation(input, ground_candidates);
    
    if (plane_coeffs.isZero()) {
        ROS_ERROR("PCA plane estimation failed, falling back to RANSAC");
        
        // Fallback to RANSAC on candidates
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        
        // Create point cloud from candidates only
        auto candidate_cloud = boost::make_shared<PointCloudXYZI>();
        for (int idx : ground_candidates) {
            candidate_cloud->push_back(input->points[idx]);
        }
        
        ransac_seg_.setInputCloud(candidate_cloud);
        ransac_seg_.segment(*inliers, *coefficients);
        
        if (inliers->indices.empty()) {
            ROS_ERROR("RANSAC also failed");
            return Eigen::Vector4f::Zero();
        }
        
        plane_coeffs = Eigen::Vector4f(coefficients->values[0], coefficients->values[1], 
                                      coefficients->values[2], coefficients->values[3]);
    }
    
    // STEP 3: STRICT point classification using the plane
    std::vector<int> final_ground_indices;
    
    for (size_t i = 0; i < input->size(); ++i) {
        const auto& point = input->points[i];
        
        // Calculate distance to plane
        float distance = std::abs(plane_coeffs[0] * point.x + plane_coeffs[1] * point.y + 
                                plane_coeffs[2] * point.z + plane_coeffs[3]) /
                        std::sqrt(plane_coeffs[0] * plane_coeffs[0] + 
                                plane_coeffs[1] * plane_coeffs[1] + 
                                plane_coeffs[2] * plane_coeffs[2]);
        
        // STRICT classification criteria
        bool is_ground = false;
        
        if (distance < params_.distance_threshold) {  // Close to plane
            // Additional Livox-inspired validation
            bool height_ok = (point.z <= params_.max_ground_height && point.z >= params_.min_ground_height);
            
            // Check if normal is sufficiently upward (CRITICAL FOR NO WALLS!)
            bool normal_ok = plane_coeffs[2] >= params_.normal_z_threshold;
            
            // Distance-based height validation
            float horizontal_dist = std::sqrt(point.x * point.x + point.y * point.y);
            bool distance_height_ok = true;
            if (horizontal_dist > 10.0f) {
                float expected_ground_height = -(plane_coeffs[0] * point.x + plane_coeffs[1] * point.y + plane_coeffs[3]) / plane_coeffs[2];
                if (std::abs(point.z - expected_ground_height) > 0.5f) {
                    distance_height_ok = false;
                }
            }
            
            is_ground = height_ok && normal_ok && distance_height_ok;
        }
        
        if (is_ground) {
            final_ground_indices.push_back(i);
        }
    }
    
    ROS_INFO("Final classification: %zu ground points from %zu total points", 
             final_ground_indices.size(), input->size());
    
    // STEP 4: Create output point clouds
    for (int idx : final_ground_indices) {
        ground_cloud->push_back(input->points[idx]);
    }
    
    // Create non-ground cloud
    std::set<int> ground_set(final_ground_indices.begin(), final_ground_indices.end());
    for (size_t i = 0; i < input->size(); ++i) {
        if (ground_set.find(i) == ground_set.end()) {
            non_ground_cloud->push_back(input->points[i]);
        }
    }
    
    // Validate the extracted ground plane
    double confidence = validateGroundPlane(plane_coeffs, ground_cloud);
    
    // Store current ground plane
    current_ground_plane_ = plane_coeffs;
    ground_plane_valid_ = (confidence > 0.5);
    
    // Update statistics
    last_stats_.ground_points = ground_cloud->size();
    last_stats_.non_ground_points = non_ground_cloud->size();
    last_stats_.validation_confidence = confidence;
    last_stats_.ground_plane_coeffs = plane_coeffs;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.extraction_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    ROS_INFO("IMPROVED ground extraction completed: %zu ground, %zu non-ground points (%.1f%% ground) in %.2f ms",
        ground_cloud->size(), non_ground_cloud->size(),
        100.0 * ground_cloud->size() / input->size(),
        last_stats_.extraction_time_ms);
    
    return plane_coeffs;
}

// SIMPLIFIED: Ground plane validation using standard PCL types
double GroundExtractor::validateGroundPlane(
    const Eigen::Vector4f& plane_coeffs,
    const PointCloudXYZI::ConstPtr& ground_points) {
    
    if (plane_coeffs.isZero() || !ground_points || ground_points->empty()) {
        return 0.0;
    }
    
    // Check plane orientation (should be roughly horizontal)
    Eigen::Vector3f normal(plane_coeffs[0], plane_coeffs[1], plane_coeffs[2]);
    normal.normalize();
    
    Eigen::Vector3f up_vector(0.0f, 0.0f, 1.0f);
    float angle_to_vertical = std::acos(std::abs(normal.dot(up_vector)));
    
    double orientation_score = 1.0 - (angle_to_vertical / (M_PI / 2));
    orientation_score = std::max(0.0, orientation_score);
    
    // Check plane height (should be reasonable ground level)
    float ground_height = -plane_coeffs[3] / plane_coeffs[2]; // z at x=0, y=0
    double height_score = 1.0;
    
    if (ground_height < params_.min_ground_height || ground_height > params_.max_ground_height) {
        height_score = 0.5;
    }
    
    // Check point density and distribution
    double density_score = std::min(1.0, ground_points->size() / 1000.0);
    
    return (orientation_score * 0.5 + height_score * 0.3 + density_score * 0.2);
}

} // namespace lidar_processing

