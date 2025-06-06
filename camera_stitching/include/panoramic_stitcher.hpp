#ifndef CAMERA_STITCHING_PANORAMIC_STITCHER_HPP
#define CAMERA_STITCHING_PANORAMIC_STITCHER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <memory>
#include <vector>
#include <chrono>
#include <mutex>
#include <cuda_runtime.h>

#include "camera_synchronizer.hpp"
#include "calibration/camera_calibrator.hpp"
#include "calibration/extrinsic_calibrator.hpp"
#include "tensor_utils.hpp"
#include "utils/memory_management.hpp"

namespace camera_stitching {

/**
 * @brief Feature matching result with geometric validation
 */
struct FeatureMatchResult {
    std::vector<cv::KeyPoint> keypoints_left;     // Detected keypoints in left image
    std::vector<cv::KeyPoint> keypoints_right;    // Detected keypoints in right image
    cv::Mat descriptors_left;                     // Feature descriptors for left image
    cv::Mat descriptors_right;                    // Feature descriptors for right image
    std::vector<cv::DMatch> matches;              // Raw feature matches
    std::vector<cv::DMatch> good_matches;         // Filtered good matches

    // Geometric validation
    cv::Mat homography;                           // Computed homography matrix
    std::vector<cv::Point2f> src_points;         // Source points for homography
    std::vector<cv::Point2f> dst_points;         // Destination points for homography
    std::vector<uchar> inlier_mask;              // RANSAC inlier mask

    // Quality metrics
    double match_confidence;                      // Overall matching confidence [0,1]
    int inlier_count;                            // Number of RANSAC inliers
    double reprojection_error;                   // Average reprojection error

    FeatureMatchResult() : match_confidence(0.0), inlier_count(0), reprojection_error(0.0) {}
};

/**
 * @brief Panoramic image result with metadata and quality metrics
 */
struct PanoramicResult {
    cv::Mat panoramic_image;                      // Final stitched panoramic image
    cv::Mat seam_mask;                           // Blending seam mask
    cv::Rect left_roi;                           // ROI of left image in panorama
    cv::Rect right_roi;                          // ROI of right image in panorama

    // Transformation parameters
    cv::Mat left_transform;                       // Transformation matrix for left image
    cv::Mat right_transform;                      // Transformation matrix for right image
    cv::Size output_size;                        // Final panorama dimensions

    // Quality assessment
    double stitching_quality;                     // Overall stitching quality [0,1]
    double overlap_percentage;                    // Percentage of image overlap
    double processing_time_ms;                    // Total processing time
    bool is_valid;                               // Result validity flag

    PanoramicResult() : stitching_quality(0.0), overlap_percentage(0.0),
                       processing_time_ms(0.0), is_valid(false) {}
};

/**
 * @brief Configuration for panoramic stitching algorithms
 */
struct StitchingConfig {
    // Feature detection parameters
    struct FeatureDetection {
        std::string detector_type = "ORB";             // Feature detector: ORB, SIFT, SURF
        int max_features = 5000;                       // Maximum features to detect
        float scale_factor = 1.2f;                     // Scale factor for ORB
        int n_levels = 8;                             // Number of pyramid levels
        int edge_threshold = 31;                       // Edge threshold for ORB
        double hessian_threshold = 400.0;             // Hessian threshold for SURF
    } feature_detection;

    // Feature matching parameters
    struct FeatureMatching {
        std::string matcher_type = "BF";               // Matcher: BF (Brute Force), FLANN
        float ratio_threshold = 0.75f;                 // Lowe's ratio test threshold
        int min_match_count = 10;                      // Minimum matches for valid homography
        double ransac_threshold = 5.0;                // RANSAC reprojection threshold
        double ransac_confidence = 0.995;             // RANSAC confidence level
        int max_iterations = 2000;                    // Maximum RANSAC iterations
    } feature_matching;

    // Image warping parameters
    struct ImageWarping {
        int interpolation_method = cv::INTER_LINEAR;   // Interpolation method
        int border_mode = cv::BORDER_REFLECT_101;      // Border handling mode
        bool enable_perspective_correction = true;     // Enable perspective correction
        double warp_sigma = 1.0;                      // Gaussian blur sigma for warping
    } image_warping;

    // Blending parameters
    struct ImageBlending {
        std::string blend_type = "multiband";          // Blending: multiband, feather, linear
        int num_bands = 5;                            // Number of bands for multiband blending
        float blend_strength = 5.0f;                  // Blending strength parameter
        bool enable_exposure_compensation = true;      // Enable exposure compensation
        bool enable_seam_finding = true;              // Enable optimal seam finding
    } image_blending;

    // Quality control
    struct QualityControl {
        double min_overlap_percentage = 10.0;         // Minimum required overlap
        double max_reprojection_error = 10.0;         // Maximum acceptable reprojection error
        int min_inlier_count = 50;                    // Minimum RANSAC inliers
        bool enable_geometric_validation = true;       // Enable geometric consistency checks
    } quality_control;

    // Performance optimization
    struct Performance {
        bool enable_cuda_acceleration = true;         // Enable CUDA acceleration
        bool enable_multithreading = true;           // Enable OpenMP multithreading
        int max_image_size = 2048;                   // Maximum processing image dimension
        bool enable_pyramid_processing = true;        // Enable image pyramid processing
        bool cache_features = true;                   // Cache features between frames
    } performance;

    // Camera calibration integration
    struct CalibrationIntegration {
        bool use_camera_calibration = true;          // Use intrinsic camera parameters
        bool enable_distortion_correction = true;    // Apply distortion correction
        std::string left_camera_config;              // Path to left camera config
        std::string right_camera_config;             // Path to right camera config
    } calibration;
};

/**
 * @brief Performance statistics for stitching monitoring
 */
struct StitchingStatistics {
    // Processing time breakdown
    double feature_detection_time_ms = 0.0;
    double feature_matching_time_ms = 0.0;
    double homography_estimation_time_ms = 0.0;
    double image_warping_time_ms = 0.0;
    double image_blending_time_ms = 0.0;
    double total_processing_time_ms = 0.0;

    // Quality metrics
    uint64_t successful_stitches = 0;
    uint64_t failed_stitches = 0;
    double average_stitch_quality = 0.0;
    double average_overlap_percentage = 0.0;

    // Feature statistics
    double average_features_detected = 0.0;
    double average_matches_found = 0.0;
    double average_inlier_ratio = 0.0;

    // Performance metrics
    double current_fps = 0.0;
    double average_memory_usage_mb = 0.0;

    std::chrono::time_point<std::chrono::high_resolution_clock> session_start;
};

/**
 * @brief GPU memory buffers for CUDA-accelerated stitching
 */
struct StitchingGpuBuffers {
    // Input image buffers
    uint8_t* d_left_image;                        // Left image device pointer
    uint8_t* d_right_image;                       // Right image device pointer

    // Processed images
    float* d_left_float;                          // Left image as normalized float
    float* d_right_float;                         // Right image as normalized float

    // Feature detection buffers
    float* d_left_features;                       // Left image feature responses
    float* d_right_features;                      // Right image feature responses
    float2* d_left_keypoints;                     // Left image keypoint coordinates
    float2* d_right_keypoints;                    // Right image keypoint coordinates
    uint8_t* d_left_descriptors;                  // Left image feature descriptors
    uint8_t* d_right_descriptors;                 // Right image feature descriptors

    // Matching and homography buffers
    int2* d_matches;                              // Feature matches (query_idx, train_idx)
    float* d_match_distances;                     // Match distances
    float* d_homography_matrix;                   // 3x3 homography matrix

    // Warping buffers
    uint8_t* d_warped_left;                       // Warped left image
    uint8_t* d_warped_right;                      // Warped right image
    float* d_warp_map_x;                         // X-coordinate warp map
    float* d_warp_map_y;                         // Y-coordinate warp map

    // Blending buffers
    float* d_blend_weights;                       // Multi-band blending weights
    uint8_t* d_panorama_output;                   // Final panoramic result

    // Buffer metadata
    int max_width, max_height, channels;          // Maximum buffer dimensions
    size_t pitch;                                 // Memory pitch for 2D arrays
    bool is_allocated;                            // Allocation status flag

    StitchingGpuBuffers() : is_allocated(false) {}
};

/**
 * @brief High-performance panoramic image stitcher with CUDA acceleration
 *
 * Professional-grade panoramic stitching implementation optimized for real-time
 * embedded applications. Features robust feature matching, geometric validation,
 * multi-band blending, and comprehensive quality assessment.
 */
class PanoramicStitcher {
public:
    explicit PanoramicStitcher(const StitchingConfig& config = StitchingConfig());
    ~PanoramicStitcher();

    // Non-copyable
    PanoramicStitcher(const PanoramicStitcher&) = delete;
    PanoramicStitcher& operator=(const PanoramicStitcher&) = delete;

    /**
     * @brief Initialize stitcher with camera calibration and GPU resources
     * @return True on successful initialization
     */
    bool initialize();

    /**
     * @brief Main stitching function for synchronized camera pair
     * @param frame_pair Synchronized stereo camera frames
     * @param result Resulting panoramic image with metadata
     * @return True if stitching was successful
     */
    bool stitch(const SynchronizedFramePair& frame_pair, PanoramicResult& result);

    /**
     * @brief Overloaded stitching function for direct image input
     * @param left_image Left camera image
     * @param right_image Right camera image
     * @param output Resulting panoramic image
     * @return True if stitching was successful
     */
    bool stitch(const cv::Mat& left_image, const cv::Mat& right_image, cv::Mat& output);

    /**
     * @brief Update stitching configuration at runtime
     * @param new_config Updated configuration parameters
     * @return True if configuration update was successful
     */
    bool updateConfiguration(const StitchingConfig& new_config);

    /**
     * @brief Load camera calibration parameters from configuration files
     * @param left_config_path Path to left camera calibration
     * @param right_config_path Path to right camera calibration
     * @return True if calibration loaded successfully
     */
    bool loadCameraCalibration(const std::string& left_config_path,
                              const std::string& right_config_path);

    // Status and monitoring
    bool isInitialized() const { return initialized_; }
    const StitchingStatistics& getStatistics() const { return statistics_; }
    void printStatistics() const;
    void resetStatistics();

    // Quality assessment
    double getLastStitchQuality() const { return last_stitch_quality_; }
    double getAverageProcessingTime() const;

    /**
     * @brief Create debug visualization showing feature matches and homography
     * @param left_image Left input image
     * @param right_image Right input image
     * @param match_result Feature matching result
     * @return Debug visualization image
     */
    cv::Mat createDebugVisualization(const cv::Mat& left_image,
                                   const cv::Mat& right_image,
                                   const FeatureMatchResult& match_result);

private:
    StitchingConfig config_;
    StitchingStatistics statistics_;

    // Feature detection and matching
    cv::Ptr<cv::FeatureDetector> feature_detector_;
    cv::Ptr<cv::DescriptorExtractor> descriptor_extractor_;
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher_;

    // Camera calibration integration
    std::unique_ptr<calibration::CameraCalibrator> left_calibrator_;
    std::unique_ptr<calibration::CameraCalibrator> right_calibrator_;
    cv::Mat left_camera_matrix_, left_dist_coeffs_;
    cv::Mat right_camera_matrix_, right_dist_coeffs_;

    // GPU acceleration components
    std::unique_ptr<lane_detection::TensorBuffer> gpu_input_buffer_;
    std::unique_ptr<lane_detection::TensorBuffer> gpu_output_buffer_;
    StitchingGpuBuffers gpu_buffers_;
    cudaStream_t cuda_stream_;

    // Memory management
    std::unique_ptr<utils::MemoryPool> memory_pool_;

    // Performance monitoring
    std::unique_ptr<lane_detection::PerformanceMonitor> performance_monitor_;

    // Feature caching for optimization
    struct FeatureCache {
        cv::Mat left_descriptors;
        cv::Mat right_descriptors;
        std::vector<cv::KeyPoint> left_keypoints;
        std::vector<cv::KeyPoint> right_keypoints;
        bool is_valid;
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;

        FeatureCache() : is_valid(false) {}
    } feature_cache_;

    // State management
    bool initialized_;
    double last_stitch_quality_;
    std::mutex processing_mutex_;

    /**
     * @brief Preprocessing pipeline with distortion correction
     */
    bool preprocessImages(const cv::Mat& left_input, const cv::Mat& right_input,
                         cv::Mat& left_output, cv::Mat& right_output);

    /**
     * @brief Detect and extract features from stereo image pair
     */
    bool detectFeatures(const cv::Mat& left_image, const cv::Mat& right_image,
                       FeatureMatchResult& result);

    /**
     * @brief Match features between stereo images with geometric validation
     */
    bool matchFeatures(FeatureMatchResult& result);

    /**
     * @brief Estimate homography with RANSAC and validate geometry
     */
    bool estimateHomography(FeatureMatchResult& result);

    /**
     * @brief Warp images to common coordinate system
     */
    bool warpImages(const cv::Mat& left_image, const cv::Mat& right_image,
                   const cv::Mat& homography, cv::Mat& warped_left, cv::Mat& warped_right,
                   cv::Rect& left_roi, cv::Rect& right_roi);

    /**
     * @brief Blend warped images with seam optimization
     */
    bool blendImages(const cv::Mat& warped_left, const cv::Mat& warped_right,
                    const cv::Rect& left_roi, const cv::Rect& right_roi,
                    cv::Mat& blended_result);

    /**
     * @brief Multi-band blending implementation
     */
    bool multibandBlend(const cv::Mat& img1, const cv::Mat& img2,
                       const cv::Mat& mask1, const cv::Mat& mask2,
                       cv::Mat& result);

    /**
     * @brief Quality assessment functions
     */
    double assessStitchingQuality(const PanoramicResult& result);
    bool validateGeometricConsistency(const FeatureMatchResult& match_result);
    double calculateOverlapPercentage(const cv::Rect& left_roi, const cv::Rect& right_roi,
                                    const cv::Size& total_size);

    /**
     * @brief GPU memory management
     */
    bool allocateGpuBuffers(int max_width, int max_height, int channels);
    void deallocateGpuBuffers();

    /**
     * @brief Update performance statistics
     */
    void updateStatistics(const PanoramicResult& result);

    /**
     * @brief Input validation
     */
    bool validateInputImages(const cv::Mat& left, const cv::Mat& right);
    bool checkCudaError(const std::string& operation);

    /**
     * @brief Logging utilities
     */
    void logError(const std::string& message);
    void logWarning(const std::string& message);
    void logInfo(const std::string& message);
};

/**
 * @brief Factory function to create panoramic stitcher from ROS parameters
 * @param nh ROS NodeHandle for parameter loading
 * @return Unique pointer to configured stitcher
 */
std::unique_ptr<PanoramicStitcher> createPanoramicStitcher(const ros::NodeHandle& nh);

/**
 * @brief Utility functions for panoramic stitching
 */
namespace stitch_utils {
    /**
     * @brief Calculate optimal output canvas size for panorama
     */
    cv::Size calculateOptimalCanvasSize(const cv::Mat& homography,
                                       const cv::Size& left_size,
                                       const cv::Size& right_size);

    /**
     * @brief Validate homography matrix for geometric consistency
     */
    bool validateHomography(const cv::Mat& homography, const cv::Size& image_size);

    /**
     * @brief Image quality assessment utilities
     */
    double calculateImageSharpness(const cv::Mat& image);
    double calculateContrastRatio(const cv::Mat& image);
    bool detectOverexposure(const cv::Mat& image, double threshold = 240.0);

    /**
     * @brief Seam finding utilities
     */
    cv::Mat findOptimalSeam(const cv::Mat& img1, const cv::Mat& img2,
                           const cv::Rect& overlap_region);

    /**
     * @brief Exposure compensation utilities
     */
    void compensateExposure(cv::Mat& img1, cv::Mat& img2, const cv::Rect& overlap_region);
}

} // namespace camera_stitching

/**
 * @brief CUDA kernel launch functions for stitching acceleration
 *
 * These C-style functions provide interfaces for launching CUDA kernels
 * from the panoramic stitcher implementation.
 */
extern "C" {

/**
 * @brief Image preprocessing kernels
 */
cudaError_t launchUndistortImageKernel(
    const uint8_t* d_input, uint8_t* d_output,
    const float* d_camera_matrix, const float* d_dist_coeffs,
    int width, int height, cudaStream_t stream);

cudaError_t launchBgrToGrayscaleKernel(
    const uint8_t* d_input, uint8_t* d_output,
    int width, int height, float sigma, cudaStream_t stream);

/**
 * @brief Feature detection kernels
 */
cudaError_t launchOrbFeatureDetectionKernel(
    const uint8_t* d_image, float2* d_keypoints, uint8_t* d_descriptors,
    int* d_keypoint_count, int width, int height, int max_features,
    float scale_factor, int n_levels, int edge_threshold, cudaStream_t stream);

cudaError_t launchHarrisCornerDetectionKernel(
    const uint8_t* d_image, float* d_corners, float2* d_keypoints,
    int* d_keypoint_count, int width, int height, float threshold,
    int max_keypoints, cudaStream_t stream);

/**
 * @brief Feature matching kernels
 */
cudaError_t launchBruteForceMatchingKernel(
    const uint8_t* d_descriptors1, const uint8_t* d_descriptors2,
    int2* d_matches, float* d_distances, int* d_match_count,
    int desc_count1, int desc_count2, int desc_length,
    float ratio_threshold, cudaStream_t stream);

/**
 * @brief Image warping kernels
 */
cudaError_t launchPerspectiveWarpKernel(
    const uint8_t* d_input, uint8_t* d_output,
    const float* d_homography, int input_width, int input_height,
    int output_width, int output_height, int channels,
    int interpolation_mode, cudaStream_t stream);

cudaError_t launchGenerateWarpMapKernel(
    float* d_warp_map_x, float* d_warp_map_y,
    const float* d_homography, int width, int height, cudaStream_t stream);

/**
 * @brief Image blending kernels
 */
cudaError_t launchMultibandBlendingKernel(
    const uint8_t* d_img1, const uint8_t* d_img2,
    const float* d_weights1, const float* d_weights2,
    uint8_t* d_result, int width, int height, int channels,
    int num_bands, float blend_strength, cudaStream_t stream);

cudaError_t launchLinearBlendingKernel(
    const uint8_t* d_img1, const uint8_t* d_img2,
    const float* d_weights, uint8_t* d_result,
    int width, int height, int channels, cudaStream_t stream);

/**
 * @brief Utility kernels
 */
cudaError_t launchHomographyRansacKernel(
    const float2* d_src_points, const float2* d_dst_points,
    float* d_homography, uchar* d_inlier_mask,
    int point_count, float threshold, float confidence,
    int max_iterations, int* d_inlier_count, cudaStream_t stream);

cudaError_t launchCalculateReprojectionErrorKernel(
    const float2* d_src_points, const float2* d_dst_points,
    const float* d_homography, float* d_errors,
    int point_count, cudaStream_t stream);

} // extern "C"

#endif // CAMERA_STITCHING_PANORAMIC_STITCHER_HPP