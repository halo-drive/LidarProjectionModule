#include "lidar_processing/ground_extraction.hpp"
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/segmentation/region_growing.h>
#include <algorithm>
#include <chrono>

namespace lidar_processing {

GroundExtractor::GroundExtractor(const GroundExtractionParams& params)
    : params_(params), ground_plane_valid_(false) {
    
    // Configure RANSAC segmentation
    ransac_seg_.setOptimizeCoefficients(true);
    ransac_seg_.setModelType(pcl::SACMODEL_PLANE);
    ransac_seg_.setMethodType(pcl::SAC_RANSAC);
    ransac_seg_.setMaxIterations(params_.max_iterations);
    ransac_seg_.setDistanceThreshold(params_.distance_threshold);
    ransac_seg_.setProbability(params_.probability);
    
    // Configure extraction
    extract_indices_.setNegative(false);
    
    // Configure normal estimation
    normal_estimator_.setRadiusSearch(params_.normal_radius);
    kdtree_.reset(new pcl::search::KdTree<PointXYZIR>());
    normal_estimator_.setSearchMethod(kdtree_);
}

Eigen::Vector4f GroundExtractor::extractGroundPlane(
    const PointCloudXYZIR::ConstPtr& input,
    PointCloudGround::Ptr& ground_cloud,
    PointCloudXYZIR::Ptr& non_ground_cloud) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Reset statistics
    last_stats_ = ExtractionStats();
    last_stats_.input_points = input->size();
    
    ROS_INFO("Extracting ground from %zu points", input->size());
    
    // Initialize output clouds
    ground_cloud.reset(new PointCloudGround());
    non_ground_cloud.reset(new PointCloudXYZIR());
    
    if (input->empty()) {
        ROS_WARN("Input cloud is empty");
        return Eigen::Vector4f::Zero();
    }
    
    // Method 1: RANSAC-based ground extraction
    PointCloudGround::Ptr ransac_ground;
    PointCloudXYZIR::Ptr ransac_non_ground;
    Eigen::Vector4f plane_coeffs = extractGroundRANSAC(input, ransac_ground, ransac_non_ground);
    
    // Method 2: Ring-based ground extraction (VLP-16 specific)
    PointCloudGround::Ptr ring_ground;
    PointCloudXYZIR::Ptr ring_non_ground;
    bool ring_success = extractGroundByRings(input, ring_ground, ring_non_ground);
    
    // Combine results from both methods
    if (ring_success && ransac_ground && ring_ground) {
        // Merge ground points from both methods
        *ground_cloud = *ransac_ground;
        
        // Add unique points from ring-based method
        for (const auto& point : ring_ground->points) {
            bool found = false;
            for (const auto& existing : ground_cloud->points) {
                float dist = std::sqrt(
                    std::pow(point.x - existing.x, 2) +
                    std::pow(point.y - existing.y, 2) +
                    std::pow(point.z - existing.z, 2)
                );
                if (dist < params_.connectivity_radius) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                ground_cloud->push_back(point);
            }
        }
        
        *non_ground_cloud = *ransac_non_ground;
    } else if (ransac_ground) {
        // Use RANSAC results only
        *ground_cloud = *ransac_ground;
        *non_ground_cloud = *ransac_non_ground;
    } else {
        ROS_ERROR("Ground extraction failed");
        return Eigen::Vector4f::Zero();
    }
    
    // Validate the extracted ground plane
    double confidence = validateGroundPlane(plane_coeffs, input);
    
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
    
    ROS_INFO("Ground extraction completed: %zu ground, %zu non-ground points (%.1f%% ground) in %.2f ms",
        ground_cloud->size(), non_ground_cloud->size(),
        100.0 * ground_cloud->size() / input->size(),
        last_stats_.extraction_time_ms);
    
    return plane_coeffs;
}

bool GroundExtractor::extractGroundByRings(
    const PointCloudXYZIR::ConstPtr& input,
    PointCloudGround::Ptr& ground_cloud,
    PointCloudXYZIR::Ptr& non_ground_cloud) {
    
    ground_cloud.reset(new PointCloudGround());
    non_ground_cloud.reset(new PointCloudXYZIR());
    
    // Organize points by rings
    std::vector<PointCloudXYZIR::Ptr> rings(VLP16_RINGS);
    for (int i = 0; i < VLP16_RINGS; ++i) {
        rings[i].reset(new PointCloudXYZIR());
    }
    
    // Distribute points into rings
    for (const auto& point : input->points) {
        if (point.ring < VLP16_RINGS) {
            rings[point.ring]->push_back(point);
        }
    }
    
    // Process each ring
    for (int ring_id = params_.start_ring; ring_id <= params_.end_ring && ring_id < VLP16_RINGS; ++ring_id) {
        if (rings[ring_id]->empty()) continue;
        
        auto ring_ground_indices = processRing(rings[ring_id], ring_id);
        
        if (!ring_ground_indices || ring_ground_indices->indices.empty()) continue;
        
        // Extract ground points from this ring
        for (int idx : ring_ground_indices->indices) {
            const auto& point = rings[ring_id]->points[idx];
            
            GroundPoint ground_point;
            ground_point.x = point.x;
            ground_point.y = point.y;
            ground_point.z = point.z;
            ground_point.confidence = 0.8f; // Ring-based confidence
            
            // Estimate normal (simplified for ring-based approach)
            ground_point.normal_x = 0.0f;
            ground_point.normal_y = 0.0f;
            ground_point.normal_z = 1.0f;
            ground_point.curvature = 0.0f;
            
            ground_cloud->push_back(ground_point);
        }
        
        // Add non-ground points
        for (size_t i = 0; i < rings[ring_id]->size(); ++i) {
            bool is_ground = std::find(ring_ground_indices->indices.begin(),
                                     ring_ground_indices->indices.end(), i) 
                           != ring_ground_indices->indices.end();
            
            if (!is_ground) {
                non_ground_cloud->push_back(rings[ring_id]->points[i]);
            }
        }
    }
    
    return !ground_cloud->empty();
}

Eigen::Vector4f GroundExtractor::extractGroundRANSAC(
    const PointCloudXYZIR::ConstPtr& input,
    PointCloudGround::Ptr& ground_cloud,
    PointCloudXYZIR::Ptr& non_ground_cloud) {
    
    ground_cloud.reset(new PointCloudGround());
    non_ground_cloud.reset(new PointCloudXYZIR());
    
    // RANSAC plane segmentation
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    
    ransac_seg_.setInputCloud(input);
    ransac_seg_.segment(*inliers, *coefficients);
    
    if (inliers->indices.empty()) {
        ROS_ERROR("RANSAC failed to find ground plane");
        return Eigen::Vector4f::Zero();
    }
    
    Eigen::Vector4f plane_coeffs(coefficients->values[0], coefficients->values[1], 
                                coefficients->values[2], coefficients->values[3]);
    
    // Estimate normals for ground points
    auto normals = estimateNormals(input);
    
    // Extract ground points with confidence calculation
    std::vector<float> confidences;
    confidences.reserve(inliers->indices.size());
    
    for (int idx : inliers->indices) {
        const auto& point = input->points[idx];
        
        // Calculate distance to plane
        float distance = std::abs(plane_coeffs[0] * point.x + plane_coeffs[1] * point.y + 
                                plane_coeffs[2] * point.z + plane_coeffs[3]) /
                        std::sqrt(plane_coeffs[0] * plane_coeffs[0] + 
                                plane_coeffs[1] * plane_coeffs[1] + 
                                plane_coeffs[2] * plane_coeffs[2]);
        
        pcl::Normal normal;
        if (idx < static_cast<int>(normals->size())) {
            normal = normals->points[idx];
        } else {
            normal.normal_x = plane_coeffs[0];
            normal.normal_y = plane_coeffs[1];
            normal.normal_z = plane_coeffs[2];
        }
        
        float confidence = calculateGroundConfidence(point, normal, distance);
        confidences.push_back(confidence);
    }
    
    // Convert to ground points
    ground_cloud = convertToGroundPoints(input, confidences, normals);
    
    // Extract non-ground points
    extract_indices_.setInputCloud(input);
    extract_indices_.setIndices(inliers);
    extract_indices_.setNegative(true);
    extract_indices_.filter(*non_ground_cloud);
    
    return plane_coeffs;
}

std::vector<PointCloudXYZIR::Ptr> GroundExtractor::segmentRadially(
    const PointCloudXYZIR::ConstPtr& input) {
    
    const int num_segments = static_cast<int>(2 * M_PI / params_.segment_distance * 10); // Approximate
    std::vector<PointCloudXYZIR::Ptr> segments(num_segments);
    
    for (int i = 0; i < num_segments; ++i) {
        segments[i].reset(new PointCloudXYZIR());
    }
    
    for (const auto& point : input->points) {
        float azimuth = std::atan2(point.y, point.x);
        if (azimuth < 0) azimuth += 2 * M_PI;
        
        int segment_id = static_cast<int>(azimuth / (2 * M_PI) * num_segments);
        segment_id = std::min(segment_id, num_segments - 1);
        
        segments[segment_id]->push_back(point);
    }
    
    // Remove segments with too few points
    segments.erase(
        std::remove_if(segments.begin(), segments.end(),
            [this](const PointCloudXYZIR::Ptr& segment) {
                return segment->size() < static_cast<size_t>(params_.min_segment_points);
            }),
        segments.end());
    
    return segments;
}

pcl::PointCloud<pcl::Normal>::Ptr GroundExtractor::estimateNormals(
    const PointCloudXYZIR::ConstPtr& input) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto normals = boost::make_shared<pcl::PointCloud<pcl::Normal>>();
    
    normal_estimator_.setInputCloud(input);
    normal_estimator_.compute(*normals);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.normal_estimation_time_ms = 
        std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return normals;
}

double GroundExtractor::validateGroundPlane(
    const Eigen::Vector4f& plane_coeffs,
    const PointCloudXYZIR::ConstPtr& ground_points) {
    
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

pcl::PointIndices::Ptr GroundExtractor::morphologicalFilter(
    const pcl::PointIndices::ConstPtr& ground_indices,
    const PointCloudXYZIR::ConstPtr& input_cloud) {
    
    if (!params_.use_morphological_filter) {
        return boost::make_shared<pcl::PointIndices>(*ground_indices);
    }
    
    // Simple morphological operations on point indices
    auto filtered_indices = boost::make_shared<pcl::PointIndices>();
    
    // For each ground point, check connectivity with neighbors
    for (int idx : ground_indices->indices) {
        const auto& point = input_cloud->points[idx];
        int neighbor_count = 0;
        
        // Count neighboring ground points
        for (int other_idx : ground_indices->indices) {
            if (idx == other_idx) continue;
            
            const auto& other_point = input_cloud->points[other_idx];
            float dist = std::sqrt(
                std::pow(point.x - other_point.x, 2) +
                std::pow(point.y - other_point.y, 2) +
                std::pow(point.z - other_point.z, 2)
            );
            
            if (dist < params_.connectivity_radius) {
                neighbor_count++;
            }
        }
        
        // Keep point if it has enough neighbors
        if (neighbor_count >= 2) {
            filtered_indices->indices.push_back(idx);
        }
    }
    
    return filtered_indices;
}

PointCloudGround::Ptr GroundExtractor::convertToGroundPoints(
    const PointCloudXYZI::ConstPtr& input,
    const std::vector<float>& ground_confidence,
    const pcl::PointCloud<pcl::Normal>::ConstPtr& normals) {
    
    auto ground_cloud = boost::make_shared<PointCloudGround>();
    
    size_t valid_count = std::min({input->size(), ground_confidence.size(), 
                                  normals ? normals->size() : input->size()});
    
    for (size_t i = 0; i < valid_count; ++i) {
        if (ground_confidence[i] > 0.5f) { // Threshold for ground classification
            GroundPoint ground_point;
            
            const auto& point = input->points[i];
            ground_point.x = point.x;
            ground_point.y = point.y;
            ground_point.z = point.z;
            ground_point.confidence = ground_confidence[i];
            
            if (normals && i < normals->size()) {
                const auto& normal = normals->points[i];
                ground_point.normal_x = normal.normal_x;
                ground_point.normal_y = normal.normal_y;
                ground_point.normal_z = normal.normal_z;
                ground_point.curvature = normal.curvature;
            } else {
                // Default normal pointing up
                ground_point.normal_x = 0.0f;
                ground_point.normal_y = 0.0f;
                ground_point.normal_z = 1.0f;
                ground_point.curvature = 0.0f;
            }
            
            ground_cloud->push_back(ground_point);
        }
    }
    
    return ground_cloud;
}

float GroundExtractor::getGroundHeightAt(float x, float y) const {
    if (!ground_plane_valid_) {
        return 0.0f;
    }
    
    // Calculate z from plane equation: ax + by + cz + d = 0
    // z = -(ax + by + d) / c
    if (std::abs(current_ground_plane_[2]) < 1e-6) {
        return 0.0f; // Avoid division by zero
    }
    
    return -(current_ground_plane_[0] * x + current_ground_plane_[1] * y + current_ground_plane_[3]) / 
           current_ground_plane_[2];
}

float GroundExtractor::classifyPoint(const PointXYZIR& point) const {
    if (!ground_plane_valid_) {
        return 0.0f;
    }
    
    // Calculate distance to ground plane
    float distance = std::abs(current_ground_plane_[0] * point.x + 
                             current_ground_plane_[1] * point.y + 
                             current_ground_plane_[2] * point.z + 
                             current_ground_plane_[3]) /
                    std::sqrt(current_ground_plane_[0] * current_ground_plane_[0] + 
                             current_ground_plane_[1] * current_ground_plane_[1] + 
                             current_ground_plane_[2] * current_ground_plane_[2]);
    
    // Convert distance to probability
    float probability = std::exp(-distance / params_.distance_threshold);
    return std::min(1.0f, probability);
}

pcl::PointIndices::Ptr GroundExtractor::processRing(
    const PointCloudXYZIR::ConstPtr& ring_points,
    int /*ring_id*/) {
    
    auto ground_indices = boost::make_shared<pcl::PointIndices>();
    
    if (ring_points->empty()) return ground_indices;
    
    // Sort points by range
    std::vector<std::pair<float, size_t>> range_indices;
    for (size_t i = 0; i < ring_points->size(); ++i) {
        range_indices.emplace_back(ring_points->points[i].range, i);
    }
    std::sort(range_indices.begin(), range_indices.end());
    
    // Find ground points using height variation
    float prev_height = ring_points->points[range_indices[0].second].z;
    ground_indices->indices.push_back(range_indices[0].second);
    
    for (size_t i = 1; i < range_indices.size(); ++i) {
        size_t idx = range_indices[i].second;
        float current_height = ring_points->points[idx].z;
        
        // Check height variation
        if (std::abs(current_height - prev_height) < params_.ring_height_threshold) {
            ground_indices->indices.push_back(idx);
            prev_height = current_height;
        } else if (current_height < prev_height) {
            // Sudden drop might still be ground
            ground_indices->indices.push_back(idx);
            prev_height = current_height;
        }
    }
    
    return ground_indices;
}

float GroundExtractor::calculateGroundConfidence(
    const PointXYZIR& point,
    const pcl::Normal& normal,
    float plane_distance) const {
    
    float confidence = 1.0f;
    
    // Distance-based confidence
    float distance_confidence = std::exp(-plane_distance / params_.distance_threshold);
    confidence *= distance_confidence;
    
    // Normal-based confidence (should point roughly upward)
    Eigen::Vector3f point_normal(normal.normal_x, normal.normal_y, normal.normal_z);
    Eigen::Vector3f up_vector(0.0f, 0.0f, 1.0f);
    
    float normal_alignment = std::abs(point_normal.normalized().dot(up_vector));
    confidence *= normal_alignment;
    
    // Height-based confidence
    if (point.z < params_.min_ground_height || point.z > params_.max_ground_height) {
        confidence *= 0.5f;
    }
    
    return std::min(1.0f, std::max(0.0f, confidence));
}

pcl::PointIndices::Ptr GroundExtractor::mergeGroundIndices(
    const pcl::PointIndices::ConstPtr& ransac_indices,
    const pcl::PointIndices::ConstPtr& ring_indices,
    const PointCloudXYZIR::ConstPtr& /*input_cloud*/) {
    
    auto merged_indices = boost::make_shared<pcl::PointIndices>();
    
    // Start with RANSAC indices
    merged_indices->indices = ransac_indices->indices;
    
    // Add unique indices from ring-based method
    for (int idx : ring_indices->indices) {
        if (std::find(merged_indices->indices.begin(), merged_indices->indices.end(), idx) == 
            merged_indices->indices.end()) {
            merged_indices->indices.push_back(idx);
        }
    }
    
    return merged_indices;
}

} // namespace lidar_processing