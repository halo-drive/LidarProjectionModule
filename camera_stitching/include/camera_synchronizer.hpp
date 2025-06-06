#ifndef CAMERA_STITCHING_CAMERA_SYNCHRONIZER_HPP
#define CAMERA_STITCHING_CAMERA_SYNCHRONIZER_HPP

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>
#include <memory>
#include <functional>
#include <chrono>
#include <deque>
#include <mutex>

#include "utils/time_sync.hpp"
#include "utils/memory_management.hpp"

namespace camera_stitching {

/**
 * @brief Synchronized camera frame pair with metadata
 */
struct SynchronizedFramePair {
    cv::Mat left_image;                    // Left camera image
    cv::Mat right_image;                   // Right camera image
    ros::Time timestamp;                   // Synchronized timestamp
    std::string left_frame_id;             // Left camera frame ID
    std::string right_frame_id;            // Right camera frame ID

    // Camera info for geometric processing
    sensor_msgs::CameraInfo left_camera_info;
    sensor_msgs::CameraInfo right_camera_info;

    // Quality metrics
    double synchronization_error_ms;       // Temporal sync error
    bool is_valid;                         // Frame pair validity flag

    SynchronizedFramePair() : synchronization_error_ms(0.0), is_valid(false) {}
};

/**
 * @brief Configuration for camera synchronization
 */
struct SynchronizationConfig {
    // Timing parameters
    double max_sync_tolerance_ms = 10.0;          // Maximum temporal offset tolerance
    double target_fps = 30.0;                     // Target synchronization rate

    // Buffer management
    int max_buffer_size = 5;                      // Maximum frame buffer size
    bool enable_frame_dropping = true;            // Enable frame dropping for performance

    // Quality control
    double min_image_quality_threshold = 0.7;     // Minimum acceptable image quality
    bool enable_adaptive_sync = true;             // Enable adaptive synchronization

    // Performance optimization
    bool enable_zero_copy = true;                 // Enable zero-copy optimizations
    bool use_exact_time_sync = false;             // Use exact vs approximate time sync

    // Topic configuration
    std::string left_image_topic = "/camera0/usb_cam_node/image_raw";
    std::string right_image_topic = "/camera1/usb_cam_node/image_raw";
    std::string left_camera_info_topic = "/camera0/usb_cam_node/camera_info";
    std::string right_camera_info_topic = "/camera1/usb_cam_node/camera_info";
};

/**
 * @brief Performance statistics for synchronization monitoring
 */
struct SynchronizationStatistics {
    uint64_t total_frames_received = 0;           // Total frames from both cameras
    uint64_t synchronized_pairs_created = 0;      // Successfully synchronized pairs
    uint64_t frames_dropped = 0;                  // Frames dropped due to sync issues

    double average_sync_error_ms = 0.0;           // Average synchronization error
    double max_sync_error_ms = 0.0;               // Maximum observed sync error
    double current_fps = 0.0;                     // Current processing rate

    std::chrono::time_point<std::chrono::high_resolution_clock> session_start;

    // Quality metrics
    double sync_success_rate = 0.0;               // Percentage of successful synchronizations
    double average_processing_latency_ms = 0.0;   // Average end-to-end latency
};

/**
 * @brief Callback function type for synchronized frame pairs
 */
using FramePairCallback = std::function<void(const SynchronizedFramePair&)>;

/**
 * @brief High-performance camera synchronizer for multi-camera stitching
 *
 * This class provides robust temporal synchronization of multiple camera streams
 * with sub-millisecond precision, optimized for real-time embedded applications.
 * Features include adaptive synchronization, frame quality validation, and
 * comprehensive performance monitoring.
 */
class CameraSynchronizer {
public:
    explicit CameraSynchronizer(const SynchronizationConfig& config = SynchronizationConfig());
    ~CameraSynchronizer();

    // Non-copyable
    CameraSynchronizer(const CameraSynchronizer&) = delete;
    CameraSynchronizer& operator=(const CameraSynchronizer&) = delete;

    /**
     * @brief Initialize the synchronizer with ROS subscribers and message filters
     * @return True on successful initialization, false otherwise
     */
    bool initialize();

    /**
     * @brief Start frame synchronization process
     * @return True if synchronization started successfully
     */
    bool start();

    /**
     * @brief Stop frame synchronization and cleanup resources
     */
    void stop();

    /**
     * @brief Register callback for synchronized frame pairs
     * @param callback Function to be called with each synchronized pair
     */
    void registerFramePairCallback(const FramePairCallback& callback);

    /**
     * @brief Update synchronization configuration at runtime
     * @param new_config Updated configuration parameters
     * @return True if configuration update was successful
     */
    bool updateConfiguration(const SynchronizationConfig& new_config);

    // Status and monitoring
    bool isInitialized() const { return initialized_; }
    bool isRunning() const { return running_; }
    const SynchronizationStatistics& getStatistics() const { return statistics_; }
    void printStatistics() const;
    void resetStatistics();

    // Quality assessment
    double getCurrentSyncQuality() const;
    double getAverageFrameRate() const;

    /**
     * @brief Manual synchronization for testing or special use cases
     * @param left_msg Left camera image message
     * @param right_msg Right camera image message
     * @return Synchronized frame pair if successful
     */
    std::optional<SynchronizedFramePair> manualSync(
        const sensor_msgs::ImageConstPtr& left_msg,
        const sensor_msgs::ImageConstPtr& right_msg);

private:
    SynchronizationConfig config_;
    SynchronizationStatistics statistics_;

    // ROS components
    ros::NodeHandle nh_;
    std::unique_ptr<image_transport::ImageTransport> it_;

    // Message filter synchronization (exact time)
    using ExactTimeSyncPolicy = message_filters::sync_policies::ExactTime<
        sensor_msgs::Image, sensor_msgs::Image>;
    using ExactTimeSynchronizer = message_filters::Synchronizer<ExactTimeSyncPolicy>;

    // Message filter synchronization (approximate time)
    using ApproximateTimeSyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image, sensor_msgs::Image>;
    using ApproximateTimeSynchronizer = message_filters::Synchronizer<ApproximateTimeSyncPolicy>;

    // Subscribers
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> left_image_sub_;
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> right_image_sub_;
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::CameraInfo>> left_info_sub_;
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::CameraInfo>> right_info_sub_;

    // Synchronizers
    std::unique_ptr<ExactTimeSynchronizer> exact_sync_;
    std::unique_ptr<ApproximateTimeSynchronizer> approximate_sync_;

    // Frame buffer management
    std::deque<SynchronizedFramePair> frame_buffer_;
    std::mutex buffer_mutex_;

    // Camera info caching
    sensor_msgs::CameraInfo cached_left_info_;
    sensor_msgs::CameraInfo cached_right_info_;
    bool left_info_received_;
    bool right_info_received_;
    std::mutex info_mutex_;

    // Callback management
    FramePairCallback user_callback_;
    std::mutex callback_mutex_;

    // State management
    bool initialized_;
    bool running_;
    std::mutex state_mutex_;

    // Performance monitoring
    std::unique_ptr<utils::PerformanceTimer> sync_timer_;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_stats_update_;

    /**
     * @brief Main synchronization callback for exact time sync
     */
    void exactTimeSyncCallback(
        const sensor_msgs::ImageConstPtr& left_msg,
        const sensor_msgs::ImageConstPtr& right_msg);

    /**
     * @brief Main synchronization callback for approximate time sync
     */
    void approximateTimeSyncCallback(
        const sensor_msgs::ImageConstPtr& left_msg,
        const sensor_msgs::ImageConstPtr& right_msg);

    /**
     * @brief Camera info callbacks
     */
    void leftCameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& info_msg);
    void rightCameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& info_msg);

    /**
     * @brief Process synchronized image pair
     */
    void processSynchronizedPair(
        const sensor_msgs::ImageConstPtr& left_msg,
        const sensor_msgs::ImageConstPtr& right_msg);

    /**
     * @brief Convert ROS image message to OpenCV matrix with error handling
     */
    bool convertRosToOpenCV(const sensor_msgs::ImageConstPtr& msg, cv::Mat& output);

    /**
     * @brief Validate frame quality metrics
     */
    bool validateFrameQuality(const cv::Mat& image);

    /**
     * @brief Calculate synchronization error between timestamps
     */
    double calculateSyncError(const ros::Time& t1, const ros::Time& t2);

    /**
     * @brief Update performance statistics
     */
    void updateStatistics(const SynchronizedFramePair& pair);

    /**
     * @brief Buffer management functions
     */
    void addToBuffer(const SynchronizedFramePair& pair);
    void pruneBuffer();

    /**
     * @brief Adaptive synchronization logic
     */
    void adaptSynchronizationParameters();

    /**
     * @brief Error handling and logging
     */
    void logError(const std::string& message);
    void logWarning(const std::string& message);
    void logInfo(const std::string& message);
};

/**
 * @brief Factory function to create camera synchronizer from ROS parameters
 * @param nh ROS NodeHandle for parameter loading
 * @return Unique pointer to configured synchronizer
 */
std::unique_ptr<CameraSynchronizer> createCameraSynchronizer(const ros::NodeHandle& nh);

/**
 * @brief Utility functions for camera synchronization
 */
namespace sync_utils {
    /**
     * @brief Calculate temporal offset between two timestamps
     */
    double calculateTemporalOffset(const ros::Time& t1, const ros::Time& t2);

    /**
     * @brief Validate camera info message consistency
     */
    bool validateCameraInfoConsistency(
        const sensor_msgs::CameraInfo& left_info,
        const sensor_msgs::CameraInfo& right_info);

    /**
     * @brief Image quality assessment functions
     */
    double assessImageSharpness(const cv::Mat& image);
    double assessImageBrightness(const cv::Mat& image);
    bool detectMotionBlur(const cv::Mat& image, double threshold = 100.0);
}

} // namespace camera_stitching

#endif // CAMERA_STITCHING_CAMERA_SYNCHRONIZER_HPP