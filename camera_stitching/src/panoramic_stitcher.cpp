#include "panoramic_stitcher.hpp"
#include <ros/package.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <numeric>

// Simple performance timer implementation
class SimplePerformanceTimer {
public:
    void start(const std::string& operation) {
        start_times_[operation] = std::chrono::high_resolution_clock::now();
    }

    void end(const std::string& operation) {
        auto it = start_times_.find(operation);
        if (it != start_times_.end()) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - it->second).count();
            total_times_[operation] += duration;
            counts_[operation]++;
        }
    }

    double getAverageTime(const std::string& operation) const {
        auto it = counts_.find(operation);
        if (it != counts_.end() && it->second > 0) {
            return static_cast<double>(total_times_.at(operation)) / it->second;
        }
        return 0.0;
    }

private:
    std::map<std::string, std::chrono::high_resolution_clock::time_point> start_times_;
    std::map<std::string, long long> total_times_;
    std::map<std::string, int> counts_;
};

// Simple memory pool placeholder
class SimpleMemoryPool {
public:
    SimpleMemoryPool(size_t size) : pool_size_(size) {}
    void* allocate(size_t size) { return malloc(size); }
    void deallocate(void* ptr) { free(ptr); }
private:
    size_t pool_size_;
};

namespace camera_stitching {

PanoramicStitcher::PanoramicStitcher(const StitchingConfig& config)
    : config_(config), initialized_(false), last_stitch_quality_(0.0), gpu_buffers_{0,0,0,false} {

    statistics_.session_start = std::chrono::high_resolution_clock::now();

    // Initialize CUDA stream
    if (cudaStreamCreate(&cuda_stream_) != cudaSuccess) {
        logError("Failed to create CUDA stream");
    }

    logInfo("PanoramicStitcher constructor completed");
}

PanoramicStitcher::~PanoramicStitcher() {
    deallocateGpuBuffers();

    if (cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
    }

    logInfo("PanoramicStitcher destructor completed");
}

bool PanoramicStitcher::initialize() {
    std::lock_guard<std::mutex> lock(processing_mutex_);

    if (initialized_) {
        logWarning("Stitcher already initialized");
        return true;
    }

    try {
        // Initialize performance monitor
        performance_monitor_ = std::make_unique<lane_detection::PerformanceMonitor>();

        // Initialize memory pool
        memory_pool_ = std::make_unique<lane_fusion::utils::MemoryPool>(
           lane_fusion::utils::MemoryType::HOST,
           lane_fusion::utils::MemoryPoolConfig{
               .initialBlockSize = 1024 * 1024,       // 1MB initial
               .growthFactor = 2,
               .maxPoolSize = 1024 * 1024 * 100,      // 100MB max
               .trackUsage = true
           });


        // Initialize feature detector based on configuration
        if (config_.feature_detection.detector_type == "ORB") {
            feature_detector_ = cv::ORB::create(
                config_.feature_detection.max_features,
                config_.feature_detection.scale_factor,
                config_.feature_detection.n_levels,
                config_.feature_detection.edge_threshold);
            descriptor_extractor_ = feature_detector_;
        } else if (config_.feature_detection.detector_type == "SIFT") {
            feature_detector_ = cv::ORB::create(config_.feature_detection.max_features);
            descriptor_extractor_ = feature_detector_;
        } else if (config_.feature_detection.detector_type == "SURF") {
            #ifdef OPENCV_XFEATURES2D_FOUND
            feature_detector_ = cv::ORB::create(config_.feature_detection.max_features);
            descriptor_extractor_ = feature_detector_;
            #else
            ROS_WARN("SURF not available, falling back to ORB");
            feature_detector_ = cv::ORB::create(config_.feature_detection.max_features);
            descriptor_extractor_ = feature_detector_;
            #endif
        } else {
            logError("Unknown feature detector type: " + config_.feature_detection.detector_type);
            return false;
        }

        // Initialize feature matcher
        if (config_.feature_matching.matcher_type == "BF") {
            if (config_.feature_detection.detector_type == "ORB") {
                descriptor_matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, true);
            } else {
                descriptor_matcher_ = cv::BFMatcher::create(cv::NORM_L2, true);
            }
        } else if (config_.feature_matching.matcher_type == "FLANN") {
            descriptor_matcher_ = cv::FlannBasedMatcher::create();
        } else {
            logError("Unknown matcher type: " + config_.feature_matching.matcher_type);
            return false;
        }

        // Allocate GPU buffers if CUDA acceleration is enabled
        if (config_.performance.enable_cuda_acceleration) {
            if (!allocateGpuBuffers(config_.performance.max_image_size,
                                   config_.performance.max_image_size, 3)) {
                logWarning("Failed to allocate GPU buffers, continuing without CUDA acceleration");
                config_.performance.enable_cuda_acceleration = false;
            }
        }

        // Initialize camera calibrators if specified
        if (config_.calibration.use_camera_calibration) {
            if (!config_.calibration.left_camera_config.empty()) {
                left_calibrator_ = std::make_unique<calibration::CameraCalibrator>();
                if (!left_calibrator_->loadCalibration(config_.calibration.left_camera_config)) {
                    logWarning("Failed to load left camera calibration");
                }
            }

            if (!config_.calibration.right_camera_config.empty()) {
                right_calibrator_ = std::make_unique<calibration::CameraCalibrator>();
                if (!right_calibrator_->loadCalibration(config_.calibration.right_camera_config)) {
                    logWarning("Failed to load right camera calibration");
                }
            }
        }

        initialized_ = true;
        logInfo("Panoramic stitcher initialized successfully");
        return true;

    } catch (const std::exception& e) {
        logError("Failed to initialize stitcher: " + std::string(e.what()));
        return false;
    }
}

bool PanoramicStitcher::stitch(const SynchronizedFramePair& frame_pair, PanoramicResult& result) {
    if (!initialized_) {
        logError("Stitcher not initialized");
        return false;
    }

    if (!frame_pair.is_valid) {
        logError("Invalid frame pair provided");
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    performance_monitor_->startTiming("total_stitching");

    try {
        // Preprocess images
        cv::Mat left_processed, right_processed;
        performance_monitor_->startTiming("preprocessing");
        bool preprocess_success = preprocessImages(frame_pair.left_image, frame_pair.right_image,
                                                 left_processed, right_processed);
        performance_monitor_->endTiming("preprocessing");

        if (!preprocess_success) {
            logError("Image preprocessing failed");
            return false;
        }

        // Detect and match features
        FeatureMatchResult match_result;
        performance_monitor_->startTiming("feature_detection");
        bool feature_success = detectFeatures(left_processed, right_processed, match_result);
        performance_monitor_->endTiming("feature_detection");

        if (!feature_success) {
            logError("Feature detection failed");
            return false;
        }

        // Match features
        performance_monitor_->startTiming("feature_matching");
        bool match_success = matchFeatures(match_result);
        performance_monitor_->endTiming("feature_matching");

        if (!match_success) {
            logError("Feature matching failed");
            return false;
        }

        // Estimate homography
        performance_monitor_->startTiming("homography_estimation");
        bool homography_success = estimateHomography(match_result);
        performance_monitor_->endTiming("homography_estimation");

        if (!homography_success) {
            logError("Homography estimation failed");
            return false;
        }

        // Warp images
        cv::Mat warped_left, warped_right;
        performance_monitor_->startTiming("image_warping");
        bool warp_success = warpImages(left_processed, right_processed, match_result.homography,
                                     warped_left, warped_right, result.left_roi, result.right_roi);
        performance_monitor_->endTiming("image_warping");

        if (!warp_success) {
            logError("Image warping failed");
            return false;
        }

        // Blend images
        performance_monitor_->startTiming("image_blending");
        bool blend_success = blendImages(warped_left, warped_right, result.left_roi, result.right_roi,
                                       result.panoramic_image);
        performance_monitor_->endTiming("image_blending");

        if (!blend_success) {
            logError("Image blending failed");
            return false;
        }

        // Set result metadata
        result.left_transform = cv::Mat::eye(3, 3, CV_32F);
        result.right_transform = match_result.homography.clone();
        result.output_size = result.panoramic_image.size();

        // Assess quality
        result.stitching_quality = assessStitchingQuality(result);
        result.overlap_percentage = calculateOverlapPercentage(result.left_roi, result.right_roi, result.output_size);

        auto end_time = std::chrono::high_resolution_clock::now();
        result.processing_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        result.is_valid = true;

        // Update statistics
        updateStatistics(result);
        last_stitch_quality_ = result.stitching_quality;

        performance_monitor_->endTiming("total_stitching");
        performance_monitor_->recordInferenceTime(result.processing_time_ms);

        logInfo("Stitching completed successfully - Quality: " + std::to_string(result.stitching_quality));
        return true;

    } catch (const std::exception& e) {
        performance_monitor_->endTiming("total_stitching");
        logError("Exception during stitching: " + std::string(e.what()));
        return false;
    }
}

bool PanoramicStitcher::stitch(const cv::Mat& left_image, const cv::Mat& right_image, cv::Mat& output) {
    PanoramicResult result;

    // Create a synthetic synchronized frame pair
    SynchronizedFramePair frame_pair;
    frame_pair.left_image = left_image.clone();
    frame_pair.right_image = right_image.clone();
    frame_pair.is_valid = true;

    bool success = stitch(frame_pair, result);
    if (success) {
        output = result.panoramic_image.clone();
    }

    return success;
}

bool PanoramicStitcher::updateConfiguration(const StitchingConfig& new_config) {
    std::lock_guard<std::mutex> lock(processing_mutex_);
    config_ = new_config;

    // Reinitialize components if necessary
    if (initialized_) {
        // Note: In a production system, you might want to selectively update
        // only the changed components rather than full reinitialization
        logInfo("Configuration updated - consider reinitialization for full effect");
    }

    return true;
}

bool PanoramicStitcher::loadCameraCalibration(const std::string& left_config_path,
                                             const std::string& right_config_path) {
    try {
        if (!left_config_path.empty()) {
            left_calibrator_ = std::make_unique<calibration::CameraCalibrator>();
            if (!left_calibrator_->loadCalibration(left_config_path)) {
                logError("Failed to load left camera calibration from: " + left_config_path);
                return false;
            }
        }

        if (!right_config_path.empty()) {
            right_calibrator_ = std::make_unique<calibration::CameraCalibrator>();
            if (!right_calibrator_->loadCalibration(right_config_path)) {
                logError("Failed to load right camera calibration from: " + right_config_path);
                return false;
            }
        }

        logInfo("Camera calibration loaded successfully");
        return true;

    } catch (const std::exception& e) {
        logError("Exception loading camera calibration: " + std::string(e.what()));
        return false;
    }
}

void PanoramicStitcher::printStatistics() const {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - statistics_.session_start).count() / 1000.0;

    ROS_INFO("=== Panoramic Stitcher Statistics ===");
    ROS_INFO("Session duration: %.2f seconds", elapsed_seconds);
    ROS_INFO("Successful stitches: %lu", statistics_.successful_stitches);
    ROS_INFO("Failed stitches: %lu", statistics_.failed_stitches);
    ROS_INFO("Average stitch quality: %.3f", statistics_.average_stitch_quality);
    ROS_INFO("Average overlap percentage: %.2f%%", statistics_.average_overlap_percentage);
    ROS_INFO("Average features detected: %.1f", statistics_.average_features_detected);
    ROS_INFO("Average matches found: %.1f", statistics_.average_matches_found);
    ROS_INFO("Average inlier ratio: %.3f", statistics_.average_inlier_ratio);
    ROS_INFO("Current FPS: %.2f", statistics_.current_fps);
    ROS_INFO("Average memory usage: %.2f MB", statistics_.average_memory_usage_mb);
    ROS_INFO("Processing time breakdown:");
    ROS_INFO("  Feature detection: %.2f ms", statistics_.feature_detection_time_ms);
    ROS_INFO("  Feature matching: %.2f ms", statistics_.feature_matching_time_ms);
    ROS_INFO("  Homography estimation: %.2f ms", statistics_.homography_estimation_time_ms);
    ROS_INFO("  Image warping: %.2f ms", statistics_.image_warping_time_ms);
    ROS_INFO("  Image blending: %.2f ms", statistics_.image_blending_time_ms);
    ROS_INFO("  Total processing: %.2f ms", statistics_.total_processing_time_ms);
    ROS_INFO("====================================");
}

void PanoramicStitcher::resetStatistics() {
    statistics_ = StitchingStatistics();
    statistics_.session_start = std::chrono::high_resolution_clock::now();
    if (performance_monitor_) {
        performance_monitor_->reset();
    }
}

double PanoramicStitcher::getAverageProcessingTime() const {
    if (statistics_.successful_stitches > 0) {
        return statistics_.total_processing_time_ms / statistics_.successful_stitches;
    }
    return 0.0;
}

cv::Mat PanoramicStitcher::createDebugVisualization(const cv::Mat& left_image,
                                                   const cv::Mat& right_image,
                                                   const FeatureMatchResult& match_result) {
    cv::Mat debug_image;

    try {
        // Draw matches between images
        cv::drawMatches(left_image, match_result.keypoints_left,
                       right_image, match_result.keypoints_right,
                       match_result.good_matches, debug_image,
                       cv::Scalar::all(-1), cv::Scalar::all(-1),
                       std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // Add text information
        std::string info = "Features: L=" + std::to_string(match_result.keypoints_left.size()) +
                          " R=" + std::to_string(match_result.keypoints_right.size()) +
                          " Matches=" + std::to_string(match_result.good_matches.size()) +
                          " Inliers=" + std::to_string(match_result.inlier_count) +
                          " Confidence=" + std::to_string(match_result.match_confidence);

        cv::putText(debug_image, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                   0.7, cv::Scalar(255, 255, 255), 2);

    } catch (const std::exception& e) {
        logError("Exception creating debug visualization: " + std::string(e.what()));
        debug_image = left_image.clone(); // Fallback to left image
    }

    return debug_image;
}

// Private method implementations

bool PanoramicStitcher::preprocessImages(const cv::Mat& left_input, const cv::Mat& right_input,
                                        cv::Mat& left_output, cv::Mat& right_output) {
    try {
        // Validate input images
        if (!validateInputImages(left_input, right_input)) {
            return false;
        }

        // Apply distortion correction if calibration is available
        if (config_.calibration.enable_distortion_correction) {
            if (left_calibrator_) {
                left_calibrator_->undistortImage(left_input, left_output);
            } else {
                left_input.copyTo(left_output);
            }

            if (right_calibrator_) {
                right_calibrator_->undistortImage(right_input, right_output);
            } else {
                right_input.copyTo(right_output);
            }
        } else {
            left_input.copyTo(left_output);
            right_input.copyTo(right_output);
        }

        // Resize images if they exceed maximum processing size
        int max_size = config_.performance.max_image_size;
        if (left_output.cols > max_size || left_output.rows > max_size) {
            double scale = static_cast<double>(max_size) / std::max(left_output.cols, left_output.rows);
            cv::resize(left_output, left_output, cv::Size(), scale, scale, cv::INTER_AREA);
            cv::resize(right_output, right_output, cv::Size(), scale, scale, cv::INTER_AREA);
        }

        return true;

    } catch (const std::exception& e) {
        logError("Exception in preprocessImages: " + std::string(e.what()));
        return false;
    }
}

bool PanoramicStitcher::detectFeatures(const cv::Mat& left_image, const cv::Mat& right_image,
                                      FeatureMatchResult& result) {
    try {
        // Detect keypoints and compute descriptors for left image
        feature_detector_->detectAndCompute(left_image, cv::noArray(),
                                          result.keypoints_left, result.descriptors_left);

        // Detect keypoints and compute descriptors for right image
        feature_detector_->detectAndCompute(right_image, cv::noArray(),
                                          result.keypoints_right, result.descriptors_right);

        // Validate results
        if (result.keypoints_left.size() < config_.feature_matching.min_match_count ||
            result.keypoints_right.size() < config_.feature_matching.min_match_count) {
            logWarning("Insufficient features detected");
            return false;
        }

        logInfo("Features detected - Left: " + std::to_string(result.keypoints_left.size()) +
               " Right: " + std::to_string(result.keypoints_right.size()));

        return true;

    } catch (const std::exception& e) {
        logError("Exception in detectFeatures: " + std::string(e.what()));
        return false;
    }
}

bool PanoramicStitcher::matchFeatures(FeatureMatchResult& result) {
    try {
        if (result.descriptors_left.empty() || result.descriptors_right.empty()) {
            logError("Empty descriptors provided for matching");
            return false;
        }

        // Perform initial matching
        std::vector<std::vector<cv::DMatch>> knn_matches;
        descriptor_matcher_->knnMatch(result.descriptors_left, result.descriptors_right, knn_matches, 2);

        // Apply Lowe's ratio test
        result.good_matches.clear();
        for (const auto& match_pair : knn_matches) {
            if (match_pair.size() == 2) {
                if (match_pair[0].distance < config_.feature_matching.ratio_threshold * match_pair[1].distance) {
                    result.good_matches.push_back(match_pair[0]);
                }
            }
        }

        // Store all matches for debugging
        result.matches = result.good_matches;

        // Validate match count
        if (result.good_matches.size() < config_.feature_matching.min_match_count) {
            logWarning("Insufficient good matches found: " + std::to_string(result.good_matches.size()));
            return false;
        }

        logInfo("Good matches found: " + std::to_string(result.good_matches.size()));
        return true;

    } catch (const std::exception& e) {
        logError("Exception in matchFeatures: " + std::string(e.what()));
        return false;
    }
}

bool PanoramicStitcher::estimateHomography(FeatureMatchResult& result) {
    try {
        if (result.good_matches.size() < 4) {
            logError("Insufficient matches for homography estimation");
            return false;
        }

        // Extract matched points
        result.src_points.clear();
        result.dst_points.clear();

        for (const auto& match : result.good_matches) {
            result.src_points.push_back(result.keypoints_left[match.queryIdx].pt);
            result.dst_points.push_back(result.keypoints_right[match.trainIdx].pt);
        }

        // Estimate homography using RANSAC
        result.homography = cv::findHomography(
            result.src_points, result.dst_points,
            cv::RANSAC, config_.feature_matching.ransac_threshold,
            result.inlier_mask, config_.feature_matching.max_iterations,
            config_.feature_matching.ransac_confidence);

        if (result.homography.empty()) {
            logError("Failed to compute homography");
            return false;
        }

        // Count inliers
        result.inlier_count = cv::countNonZero(result.inlier_mask);

        // Validate homography quality
        if (result.inlier_count < config_.quality_control.min_inlier_count) {
            logWarning("Insufficient inliers: " + std::to_string(result.inlier_count));
            return false;
        }

        // Calculate reprojection error
        std::vector<cv::Point2f> reprojected_points;
        cv::perspectiveTransform(result.src_points, reprojected_points, result.homography);

        double total_error = 0.0;
        int valid_points = 0;
        for (size_t i = 0; i < result.dst_points.size(); ++i) {
            if (result.inlier_mask[i]) {
                double error = cv::norm(result.dst_points[i] - reprojected_points[i]);
                total_error += error;
                valid_points++;
            }
        }

        result.reprojection_error = (valid_points > 0) ? total_error / valid_points : 0.0;

        // Validate reprojection error
        if (result.reprojection_error > config_.quality_control.max_reprojection_error) {
            logWarning("High reprojection error: " + std::to_string(result.reprojection_error));
            return false;
        }

        // Calculate match confidence
        double inlier_ratio = static_cast<double>(result.inlier_count) / result.good_matches.size();
        double error_quality = std::max(0.0, 1.0 - (result.reprojection_error / config_.quality_control.max_reprojection_error));
        result.match_confidence = inlier_ratio * error_quality;

        // Validate geometric consistency if enabled
        if (config_.quality_control.enable_geometric_validation) {
            if (!validateGeometricConsistency(result)) {
                logWarning("Geometric consistency validation failed");
                return false;
            }
        }

        logInfo("Homography estimation successful - Inliers: " + std::to_string(result.inlier_count) +
               " Error: " + std::to_string(result.reprojection_error) +
               " Confidence: " + std::to_string(result.match_confidence));

        return true;

    } catch (const std::exception& e) {
        logError("Exception in estimateHomography: " + std::string(e.what()));
        return false;
    }
}

bool PanoramicStitcher::warpImages(const cv::Mat& left_image, const cv::Mat& right_image,
                                  const cv::Mat& homography, cv::Mat& warped_left, cv::Mat& warped_right,
                                  cv::Rect& left_roi, cv::Rect& right_roi) {
    try {
        // Calculate canvas size
        cv::Size canvas_size = stitch_utils::calculateOptimalCanvasSize(homography, left_image.size(), right_image.size());

        // Calculate transformation matrices for proper alignment
        cv::Mat left_transform = cv::Mat::eye(3, 3, CV_32F);
        cv::Mat right_transform = homography.clone();

        // Adjust transforms to ensure positive coordinates
        std::vector<cv::Point2f> corners_left = {
            cv::Point2f(0, 0), cv::Point2f(left_image.cols, 0),
            cv::Point2f(left_image.cols, left_image.rows), cv::Point2f(0, left_image.rows)
        };

        std::vector<cv::Point2f> corners_right;
        cv::perspectiveTransform(corners_left, corners_right, homography);

        // Find minimum coordinates to adjust offset
        float min_x = 0, min_y = 0;
        for (const auto& corner : corners_right) {
            min_x = std::min(min_x, corner.x);
            min_y = std::min(min_y, corner.y);
        }

        // Create offset transformation
        cv::Mat offset_transform = cv::Mat::eye(3, 3, CV_32F);
        offset_transform.at<float>(0, 2) = -min_x;
        offset_transform.at<float>(1, 2) = -min_y;

        // Apply offset to transformations
        left_transform = offset_transform * left_transform;
        right_transform = offset_transform * right_transform;

        // Warp images
        cv::warpPerspective(left_image, warped_left, left_transform, canvas_size,
                           config_.image_warping.interpolation_method,
                           config_.image_warping.border_mode);

        cv::warpPerspective(right_image, warped_right, right_transform, canvas_size,
                           config_.image_warping.interpolation_method,
                           config_.image_warping.border_mode);

        // Calculate ROIs
        std::vector<cv::Point2f> left_corners_warped;
        cv::perspectiveTransform(corners_left, left_corners_warped, left_transform);
        left_roi = cv::boundingRect(left_corners_warped);

        std::vector<cv::Point2f> right_corners_warped;
        cv::perspectiveTransform(corners_left, right_corners_warped, right_transform);
        right_roi = cv::boundingRect(right_corners_warped);

        // Clamp ROIs to canvas bounds
        left_roi &= cv::Rect(0, 0, canvas_size.width, canvas_size.height);
        right_roi &= cv::Rect(0, 0, canvas_size.width, canvas_size.height);

        return true;

    } catch (const std::exception& e) {
        logError("Exception in warpImages: " + std::string(e.what()));
        return false;
    }
}

bool PanoramicStitcher::blendImages(const cv::Mat& warped_left, const cv::Mat& warped_right,
                                   const cv::Rect& left_roi, const cv::Rect& right_roi,
                                   cv::Mat& blended_result) {
    try {
        // Initialize result image
        blended_result = cv::Mat::zeros(warped_left.size(), warped_left.type());

        // Create masks for valid regions
        cv::Mat left_mask = cv::Mat::zeros(warped_left.size(), CV_8UC1);
        cv::Mat right_mask = cv::Mat::zeros(warped_right.size(), CV_8UC1);

        left_mask(left_roi) = 255;
        right_mask(right_roi) = 255;

        // Apply simple linear blending in overlap region
        cv::Rect overlap_rect = left_roi & right_roi;

        if (overlap_rect.area() > 0) {
            // Multi-band blending if enabled
            if (config_.image_blending.blend_type == "multiband") {
                cv::Mat mask1 = cv::Mat::zeros(warped_left.size(), CV_8UC1);
                cv::Mat mask2 = cv::Mat::zeros(warped_right.size(), CV_8UC1);
                mask1(left_roi) = 255;
                mask2(right_roi) = 255;

                if (!multibandBlend(warped_left, warped_right, mask1, mask2, blended_result)) {
                    logWarning("Multi-band blending failed, falling back to linear blending");
                    config_.image_blending.blend_type = "linear";
                }
            }

            // Linear blending fallback
            if (config_.image_blending.blend_type == "linear" || config_.image_blending.blend_type == "feather") {
                // Copy non-overlapping regions
                warped_left.copyTo(blended_result, left_mask & ~right_mask);
                warped_right.copyTo(blended_result, right_mask & ~left_mask);

                // Blend overlap region
                cv::Mat overlap_mask = left_mask & right_mask;
                cv::Mat left_overlap, right_overlap;
                warped_left.copyTo(left_overlap, overlap_mask);
                warped_right.copyTo(right_overlap, overlap_mask);

                cv::Mat blended_overlap;
                cv::addWeighted(left_overlap, 0.5, right_overlap, 0.5, 0, blended_overlap);
                blended_overlap.copyTo(blended_result, overlap_mask);
            }
        } else {
            // No overlap - simple copy
            warped_left.copyTo(blended_result, left_mask);
            warped_right.copyTo(blended_result, right_mask);
        }

        return true;

    } catch (const std::exception& e) {
        logError("Exception in blendImages: " + std::string(e.what()));
        return false;
    }
}

bool PanoramicStitcher::multibandBlend(const cv::Mat& img1, const cv::Mat& img2,
                                      const cv::Mat& mask1, const cv::Mat& mask2,
                                      cv::Mat& result) {
    try {
        // Simplified multi-band blending implementation
        // For production use, consider using OpenCV's detail::MultiBandBlender

        cv::detail::MultiBandBlender blender;
        blender.setNumBands(config_.image_blending.num_bands);

        // Prepare blender
        std::vector<cv::Point> corners = {cv::Point(0, 0), cv::Point(0, 0)};
        std::vector<cv::Size> sizes = {img1.size(), img2.size()};

        cv::Rect result_roi;
        result_roi.x = std::min(corners[0].x, corners[1].x);
        result_roi.y = std::min(corners[0].y, corners[1].y);
        result_roi.width = std::max(corners[0].x + sizes[0].width,
                                   corners[1].x + sizes[1].width) - result_roi.x;
        result_roi.height = std::max(corners[0].y + sizes[0].height,
                                    corners[1].y + sizes[1].height) - result_roi.y;

        // Feed images to blender
        cv::Mat img1_s, img2_s;
        img1.convertTo(img1_s, CV_16S);
        img2.convertTo(img2_s, CV_16S);

        blender.feed(img1_s, mask1, corners[0]);
        blender.feed(img2_s, mask2, corners[1]);

        // Blend and retrieve result
        cv::Mat result_s, result_mask;
        blender.blend(result_s, result_mask);
        result_s.convertTo(result, CV_8U);

        return true;

    } catch (const std::exception& e) {
        logError("Exception in multibandBlend: " + std::string(e.what()));
        return false;
    }
}

double PanoramicStitcher::assessStitchingQuality(const PanoramicResult& result) {
    try {
        // Quality assessment based on multiple factors
        double quality = 1.0;

        // Factor 1: Overlap percentage (optimal around 20-40%)
        double overlap_factor = 1.0;
        if (result.overlap_percentage < 10.0) {
            overlap_factor = result.overlap_percentage / 10.0;
        } else if (result.overlap_percentage > 50.0) {
            overlap_factor = std::max(0.5, 1.0 - (result.overlap_percentage - 50.0) / 50.0);
        }

        // Factor 2: Processing time efficiency
        double time_factor = std::min(1.0, 100.0 / result.processing_time_ms);

        // Factor 3: Image completeness (non-zero pixels ratio)
        int total_pixels = result.panoramic_image.rows * result.panoramic_image.cols;
        int non_zero_pixels = cv::countNonZero(result.panoramic_image);
        double completeness_factor = static_cast<double>(non_zero_pixels) / total_pixels;

        // Combine factors
        quality = overlap_factor * 0.4 + time_factor * 0.2 + completeness_factor * 0.4;

        return std::max(0.0, std::min(1.0, quality));

    } catch (const std::exception& e) {
        logError("Exception in assessStitchingQuality: " + std::string(e.what()));
        return 0.0;
    }
}

bool PanoramicStitcher::validateGeometricConsistency(const FeatureMatchResult& match_result) {
    // Validate homography matrix properties
    return stitch_utils::validateHomography(match_result.homography, cv::Size(640, 480));
}

double PanoramicStitcher::calculateOverlapPercentage(const cv::Rect& left_roi, const cv::Rect& right_roi,
                                                    const cv::Size& total_size) {
    cv::Rect overlap = left_roi & right_roi;
    if (overlap.area() <= 0) {
        return 0.0;
    }

    double total_area = left_roi.area() + right_roi.area() - overlap.area();
    return (static_cast<double>(overlap.area()) / total_area) * 100.0;
}

bool PanoramicStitcher::allocateGpuBuffers(int max_width, int max_height, int channels) {
    try {
        size_t image_size = max_width * max_height * channels;
        size_t float_image_size = max_width * max_height * channels * sizeof(float);

        // Allocate input buffer for image data
        gpu_input_buffer_ = std::make_unique<lane_detection::TensorBuffer>(
            image_size, nvinfer1::DataType::kFLOAT);

        // Allocate output buffer for processed results
        gpu_output_buffer_ = std::make_unique<lane_detection::TensorBuffer>(
            float_image_size * 2, nvinfer1::DataType::kFLOAT);

        // Initialize GPU buffers structure
        gpu_buffers_.max_width = max_width;
        gpu_buffers_.max_height = max_height;
        gpu_buffers_.channels = channels;
        gpu_buffers_.is_allocated = true;

        logInfo("GPU buffers allocated successfully");
        return true;

    } catch (const std::exception& e) {
        logError("Failed to allocate GPU buffers: " + std::string(e.what()));
        return false;
    }
}

void PanoramicStitcher::deallocateGpuBuffers() {
    gpu_input_buffer_.reset();
    gpu_output_buffer_.reset();
    gpu_buffers_.is_allocated = false;
    logInfo("GPU biffers deallocated");
}

void PanoramicStitcher::updateStatistics(const PanoramicResult& result) {
    if (result.is_valid) {
        statistics_.successful_stitches++;

        // Update quality metrics
        statistics_.average_stitch_quality =
            (statistics_.average_stitch_quality * (statistics_.successful_stitches - 1) + result.stitching_quality)
            / statistics_.successful_stitches;

        statistics_.average_overlap_percentage =
            (statistics_.average_overlap_percentage * (statistics_.successful_stitches - 1) + result.overlap_percentage)
            / statistics_.successful_stitches;

        // Update timing statistics
        statistics_.total_processing_time_ms =
            (statistics_.total_processing_time_ms * (statistics_.successful_stitches - 1) + result.processing_time_ms)
            / statistics_.successful_stitches;

        // Calculate current FPS
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - statistics_.session_start).count() / 1000.0;

        if (elapsed_seconds > 0.0) {
            statistics_.current_fps = static_cast<double>(statistics_.successful_stitches) / elapsed_seconds;
        }
    } else {
        statistics_.failed_stitches++;
    }
}

bool PanoramicStitcher::validateInputImages(const cv::Mat& left, const cv::Mat& right) {
    if (left.empty() || right.empty()) {
        logError("Empty input images");
        return false;
    }

    if (left.size() != right.size()) {
        logWarning("Input images have different sizes");
        // Allow different sizes but log warning
    }

    if (left.type() != right.type()) {
        logError("Input images have different types");
        return false;
    }

    return true;
}

bool PanoramicStitcher::checkCudaError(const std::string& operation) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        logError("CUDA error in " + operation + ": " + std::string(cudaGetErrorString(error)));
        return false;
    }
    return true;
}

void PanoramicStitcher::logError(const std::string& message) {
    ROS_ERROR("[PanoramicStitcher] %s", message.c_str());
}

void PanoramicStitcher::logWarning(const std::string& message) {
    ROS_WARN("[PanoramicStitcher] %s", message.c_str());
}

void PanoramicStitcher::logInfo(const std::string& message) {
    ROS_INFO("[PanoramicStitcher] %s", message.c_str());
}

// Factory function
std::unique_ptr<PanoramicStitcher> createPanoramicStitcher(const ros::NodeHandle& nh) {
    StitchingConfig config;

    // Load feature detection parameters
    nh.param<std::string>("stitch/feature_detector", config.feature_detection.detector_type, "ORB");
    nh.param<int>("stitch/max_features", config.feature_detection.max_features, 5000);
    nh.param<float>("stitch/scale_factor", config.feature_detection.scale_factor, 1.2f);
    nh.param<int>("stitch/n_levels", config.feature_detection.n_levels, 8);
    nh.param<int>("stitch/edge_threshold", config.feature_detection.edge_threshold, 31);
    nh.param<double>("stitch/hessian_threshold", config.feature_detection.hessian_threshold, 400.0);

    // Load feature matching parameters
    nh.param<std::string>("stitch/matcher_type", config.feature_matching.matcher_type, "BF");
    nh.param<float>("stitch/ratio_threshold", config.feature_matching.ratio_threshold, 0.75f);
    nh.param<int>("stitch/min_match_count", config.feature_matching.min_match_count, 10);
    nh.param<double>("stitch/ransac_threshold", config.feature_matching.ransac_threshold, 5.0);
    nh.param<double>("stitch/ransac_confidence", config.feature_matching.ransac_confidence, 0.995);
    nh.param<int>("stitch/max_iterations", config.feature_matching.max_iterations, 2000);

    // Load blending parameters
    nh.param<std::string>("stitch/blend_type", config.image_blending.blend_type, "multiband");
    nh.param<int>("stitch/num_bands", config.image_blending.num_bands, 5);
    nh.param<float>("stitch/blend_strength", config.image_blending.blend_strength, 5.0f);
    nh.param<bool>("stitch/enable_exposure_compensation", config.image_blending.enable_exposure_compensation, true);
    nh.param<bool>("stitch/enable_seam_finding", config.image_blending.enable_seam_finding, true);

    // Load quality control parameters
    nh.param<double>("stitch/min_overlap_percentage", config.quality_control.min_overlap_percentage, 10.0);
    nh.param<double>("stitch/max_reprojection_error", config.quality_control.max_reprojection_error, 10.0);
    nh.param<int>("stitch/min_inlier_count", config.quality_control.min_inlier_count, 50);
    nh.param<bool>("stitch/enable_geometric_validation", config.quality_control.enable_geometric_validation, true);

    // Load performance parameters
    nh.param<bool>("stitch/enable_cuda_acceleration", config.performance.enable_cuda_acceleration, true);
    nh.param<bool>("stitch/enable_multithreading", config.performance.enable_multithreading, true);
    nh.param<int>("stitch/max_image_size", config.performance.max_image_size, 2048);
    nh.param<bool>("stitch/enable_pyramid_processing", config.performance.enable_pyramid_processing, true);
    nh.param<bool>("stitch/cache_features", config.performance.cache_features, true);

    // Load calibration parameters
    nh.param<bool>("stitch/use_camera_calibration", config.calibration.use_camera_calibration, true);
    nh.param<bool>("stitch/enable_distortion_correction", config.calibration.enable_distortion_correction, true);
    nh.param<std::string>("stitch/left_camera_config", config.calibration.left_camera_config, "");
    nh.param<std::string>("stitch/right_camera_config", config.calibration.right_camera_config, "");

    return std::make_unique<PanoramicStitcher>(config);
}

// Utility functions implementation
namespace stitch_utils {

    cv::Size calculateOptimalCanvasSize(const cv::Mat& homography,
                                       const cv::Size& left_size,
                                       const cv::Size& right_size) {
        // Calculate corners of right image after transformation
        std::vector<cv::Point2f> corners = {
            cv::Point2f(0, 0), cv::Point2f(right_size.width, 0),
            cv::Point2f(right_size.width, right_size.height), cv::Point2f(0, right_size.height)
        };

        std::vector<cv::Point2f> transformed_corners;
        cv::perspectiveTransform(corners, transformed_corners, homography);

        // Find bounding box
        float min_x = 0, max_x = left_size.width;
        float min_y = 0, max_y = left_size.height;

        for (const auto& corner : transformed_corners) {
            min_x = std::min(min_x, corner.x);
            max_x = std::max(max_x, corner.x);
            min_y = std::min(min_y, corner.y);
            max_y = std::max(max_y, corner.y);
        }

        int canvas_width = static_cast<int>(std::ceil(max_x - min_x));
        int canvas_height = static_cast<int>(std::ceil(max_y - min_y));

        return cv::Size(canvas_width, canvas_height);
    }

    bool validateHomography(const cv::Mat& homography, const cv::Size& image_size) {
        if (homography.empty() || homography.size() != cv::Size(3, 3)) {
            return false;
        }

        // Check determinant (should not be zero)
        double det = cv::determinant(homography);
        if (std::abs(det) < 1e-6) {
            return false;
        }

        // Check corner transformation for reasonable bounds
        std::vector<cv::Point2f> corners = {
            cv::Point2f(0, 0), cv::Point2f(image_size.width, 0),
            cv::Point2f(image_size.width, image_size.height), cv::Point2f(0, image_size.height)
        };

        std::vector<cv::Point2f> transformed_corners;
        cv::perspectiveTransform(corners, transformed_corners, homography);

        // Check if transformation is reasonable (not too extreme)
        for (const auto& corner : transformed_corners) {
            if (std::abs(corner.x) > image_size.width * 10 ||
                std::abs(corner.y) > image_size.height * 10) {
                return false;
            }
        }

        return true;
    }

    double calculateImageSharpness(const cv::Mat& image) {
        if (image.empty()) return 0.0;

        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }

        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_64F);
        cv::Scalar mean, stddev;
        cv::meanStdDev(laplacian, mean, stddev);

        return stddev.val[0] * stddev.val[0];
    }

    double calculateContrastRatio(const cv::Mat& image) {
        if (image.empty()) return 0.0;

        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }

        double min_val, max_val;
        cv::minMaxLoc(gray, &min_val, &max_val);

        return (max_val > 0) ? max_val / (min_val + 1) : 0.0;
    }

    bool detectOverexposure(const cv::Mat& image, double threshold) {
        if (image.empty()) return true;

        cv::Mat mask;
        cv::threshold(image, mask, threshold, 255, cv::THRESH_BINARY);

        int overexposed_pixels = cv::countNonZero(mask);
        int total_pixels = image.rows * image.cols;

        double overexposed_ratio = static_cast<double>(overexposed_pixels) / total_pixels;
        return overexposed_ratio > 0.1; // 10% threshold
    }

    cv::Mat findOptimalSeam(const cv::Mat& img1, const cv::Mat& img2,
                           const cv::Rect& overlap_region) {
        // Simplified seam finding - return center line
        cv::Mat seam = cv::Mat::zeros(img1.size(), CV_8UC1);
        cv::Point center(overlap_region.x + overlap_region.width / 2, overlap_region.y);
        cv::Point end(overlap_region.x + overlap_region.width / 2, overlap_region.y + overlap_region.height);
        cv::line(seam, center, end, cv::Scalar(255), 1);
        return seam;
    }

    void compensateExposure(cv::Mat& img1, cv::Mat& img2, const cv::Rect& overlap_region) {
        if (overlap_region.area() <= 0) return;

        cv::Scalar mean1 = cv::mean(img1(overlap_region));
        cv::Scalar mean2 = cv::mean(img2(overlap_region));

        // Simple gain adjustment
        double gain_ratio = (mean1[0] + mean1[1] + mean1[2]) / (mean2[0] + mean2[1] + mean2[2] + 1e-6);

        if (gain_ratio > 1.5 || gain_ratio < 0.67) {
            img2 *= std::min(2.0, std::max(0.5, gain_ratio));
        }
    }
}

} // namespace camera_stitching