#include "camera_synchronizer.hpp"
#include <ros/package.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <numeric>

namespace camera_stitching {

CameraSynchronizer::CameraSynchronizer(const SynchronizationConfig& config)
    : config_(config), nh_(""),
      left_info_received_(false), right_info_received_(false),
      initialized_(false), running_(false) {

    statistics_.session_start = std::chrono::high_resolution_clock::now();

    // Initialize image transport
    it_ = std::make_unique<image_transport::ImageTransport>(nh_);

    logInfo("CameraSynchronizer constructor completed");
}

CameraSynchronizer::~CameraSynchronizer() {
    stop();
    logInfo("CameraSynchronizer destructor completed");
}

bool CameraSynchronizer::initialize() {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (initialized_) {
        logWarning("Synchronizer already initialized");
        return true;
    }

    try {
        // Initialize performance timer
        sync_timer_ = std::make_unique<utils::PerformanceTimer>();

        // Create message filter subscribers
        left_image_sub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::Image>>(
            nh_, config_.left_image_topic, 10);
        right_image_sub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::Image>>(
            nh_, config_.right_image_topic, 10);

        // Create camera info subscribers
        left_info_sub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::CameraInfo>>(
            nh_, config_.left_camera_info_topic, 10);
        right_info_sub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::CameraInfo>>(
            nh_, config_.right_camera_info_topic, 10);

        // Setup camera info callbacks
        left_info_sub_->registerCallback(
            boost::bind(&CameraSynchronizer::leftCameraInfoCallback, this, _1));
        right_info_sub_->registerCallback(
            boost::bind(&CameraSynchronizer::rightCameraInfoCallback, this, _1));

        // Create synchronizers based on configuration
        if (config_.use_exact_time_sync) {
            exact_sync_ = std::make_unique<ExactTimeSynchronizer>(
                ExactTimeSyncPolicy(10), *left_image_sub_, *right_image_sub_);
            exact_sync_->registerCallback(
                boost::bind(&CameraSynchronizer::exactTimeSyncCallback, this, _1, _2));
            logInfo("Initialized with exact time synchronization");
        } else {
            approximate_sync_ = std::make_unique<ApproximateTimeSynchronizer>(
                ApproximateTimeSyncPolicy(10), *left_image_sub_, *right_image_sub_);
            approximate_sync_->setMaxIntervalDuration(
                ros::Duration(config_.max_sync_tolerance_ms / 1000.0));
            approximate_sync_->registerCallback(
                boost::bind(&CameraSynchronizer::approximateTimeSyncCallback, this, _1, _2));
            logInfo("Initialized with approximate time synchronization");
        }

        // Initialize frame buffer
        frame_buffer_.clear();

        initialized_ = true;
        last_stats_update_ = std::chrono::high_resolution_clock::now();

        logInfo("Camera synchronizer initialized successfully");
        return true;

    } catch (const std::exception& e) {
        logError("Failed to initialize synchronizer: " + std::string(e.what()));
        return false;
    }
}

bool CameraSynchronizer::start() {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (!initialized_) {
        logError("Cannot start synchronizer: not initialized");
        return false;
    }

    if (running_) {
        logWarning("Synchronizer already running");
        return true;
    }

    running_ = true;
    statistics_.session_start = std::chrono::high_resolution_clock::now();

    logInfo("Camera synchronizer started");
    return true;
}

void CameraSynchronizer::stop() {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (!running_) {
        return;
    }

    running_ = false;

    // Clear buffers
    {
        std::lock_guard<std::mutex> buffer_lock(buffer_mutex_);
        frame_buffer_.clear();
    }

    logInfo("Camera synchronizer stopped");
}

void CameraSynchronizer::registerFramePairCallback(const FramePairCallback& callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    user_callback_ = callback;
}

bool CameraSynchronizer::updateConfiguration(const SynchronizationConfig& new_config) {
    std::lock_guard<std::mutex> lock(state_mutex_);

    // Validate new configuration
    if (new_config.max_sync_tolerance_ms <= 0.0 ||
        new_config.target_fps <= 0.0 ||
        new_config.max_buffer_size <= 0) {
        logError("Invalid configuration parameters");
        return false;
    }

    config_ = new_config;

    // Update synchronizer parameters if running
    if (running_ && approximate_sync_) {
        approximate_sync_->setMaxIntervalDuration(
            ros::Duration(config_.max_sync_tolerance_ms / 1000.0));
    }

    logInfo("Configuration updated successfully");
    return true;
}

void CameraSynchronizer::printStatistics() const {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - statistics_.session_start).count() / 1000.0;

    ROS_INFO("=== Camera Synchronizer Statistics ===");
    ROS_INFO("Session duration: %.2f seconds", elapsed_seconds);
    ROS_INFO("Total frames received: %lu", statistics_.total_frames_received);
    ROS_INFO("Synchronized pairs created: %lu", statistics_.synchronized_pairs_created);
    ROS_INFO("Frames dropped: %lu", statistics_.frames_dropped);
    ROS_INFO("Average sync error: %.3f ms", statistics_.average_sync_error_ms);
    ROS_INFO("Max sync error: %.3f ms", statistics_.max_sync_error_ms);
    ROS_INFO("Current FPS: %.2f", statistics_.current_fps);
    ROS_INFO("Sync success rate: %.2f%%", statistics_.sync_success_rate * 100.0);
    ROS_INFO("Average processing latency: %.3f ms", statistics_.average_processing_latency_ms);
    ROS_INFO("=====================================");
}

void CameraSynchronizer::resetStatistics() {
    statistics_ = SynchronizationStatistics();
    statistics_.session_start = std::chrono::high_resolution_clock::now();
    last_stats_update_ = statistics_.session_start;
}

double CameraSynchronizer::getCurrentSyncQuality() const {
    if (statistics_.synchronized_pairs_created == 0) {
        return 0.0;
    }

    // Quality based on sync success rate and average error
    double error_quality = std::max(0.0, 1.0 - (statistics_.average_sync_error_ms / config_.max_sync_tolerance_ms));
    return statistics_.sync_success_rate * error_quality;
}

double CameraSynchronizer::getAverageFrameRate() const {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - statistics_.session_start).count() / 1000.0;

    if (elapsed_seconds <= 0.0) {
        return 0.0;
    }

    return static_cast<double>(statistics_.synchronized_pairs_created) / elapsed_seconds;
}

std::optional<SynchronizedFramePair> CameraSynchronizer::manualSync(
    const sensor_msgs::ImageConstPtr& left_msg,
    const sensor_msgs::ImageConstPtr& right_msg) {

    if (!left_msg || !right_msg) {
        logError("Invalid input messages for manual sync");
        return std::nullopt;
    }

    SynchronizedFramePair pair;

    // Convert images
    if (!convertRosToOpenCV(left_msg, pair.left_image) ||
        !convertRosToOpenCV(right_msg, pair.right_image)) {
        logError("Failed to convert ROS messages to OpenCV images");
        return std::nullopt;
    }

    // Calculate sync error
    pair.synchronization_error_ms = calculateSyncError(left_msg->header.stamp, right_msg->header.stamp);

    // Set metadata
    pair.timestamp = std::max(left_msg->header.stamp, right_msg->header.stamp);
    pair.left_frame_id = left_msg->header.frame_id;
    pair.right_frame_id = right_msg->header.frame_id;

    // Validate quality
    pair.is_valid = validateFrameQuality(pair.left_image) &&
                   validateFrameQuality(pair.right_image) &&
                   pair.synchronization_error_ms <= config_.max_sync_tolerance_ms;

    // Set camera info if available
    {
        std::lock_guard<std::mutex> lock(info_mutex_);
        if (left_info_received_) pair.left_camera_info = cached_left_info_;
        if (right_info_received_) pair.right_camera_info = cached_right_info_;
    }

    return pair;
}

void CameraSynchronizer::exactTimeSyncCallback(
    const sensor_msgs::ImageConstPtr& left_msg,
    const sensor_msgs::ImageConstPtr& right_msg) {

    if (!running_) return;

    sync_timer_->start("exact_sync_callback");

    processSynchronizedPair(left_msg, right_msg);

    sync_timer_->end("exact_sync_callback");
}

void CameraSynchronizer::approximateTimeSyncCallback(
    const sensor_msgs::ImageConstPtr& left_msg,
    const sensor_msgs::ImageConstPtr& right_msg) {

    if (!running_) return;

    sync_timer_->start("approximate_sync_callback");

    processSynchronizedPair(left_msg, right_msg);

    sync_timer_->end("approximate_sync_callback");
}

void CameraSynchronizer::leftCameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& info_msg) {
    std::lock_guard<std::mutex> lock(info_mutex_);
    cached_left_info_ = *info_msg;
    left_info_received_ = true;
}

void CameraSynchronizer::rightCameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& info_msg) {
    std::lock_guard<std::mutex> lock(info_mutex_);
    cached_right_info_ = *info_msg;
    right_info_received_ = true;
}

void CameraSynchronizer::processSynchronizedPair(
    const sensor_msgs::ImageConstPtr& left_msg,
    const sensor_msgs::ImageConstPtr& right_msg) {

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        SynchronizedFramePair pair;

        // Convert ROS images to OpenCV
        if (!convertRosToOpenCV(left_msg, pair.left_image) ||
            !convertRosToOpenCV(right_msg, pair.right_image)) {
            statistics_.frames_dropped++;
            logWarning("Failed to convert ROS images to OpenCV format");
            return;
        }

        // Calculate synchronization metrics
        pair.synchronization_error_ms = calculateSyncError(left_msg->header.stamp, right_msg->header.stamp);
        pair.timestamp = std::max(left_msg->header.stamp, right_msg->header.stamp);
        pair.left_frame_id = left_msg->header.frame_id;
        pair.right_frame_id = right_msg->header.frame_id;

        // Validate frame quality
        bool left_quality_ok = validateFrameQuality(pair.left_image);
        bool right_quality_ok = validateFrameQuality(pair.right_image);
        bool sync_error_ok = pair.synchronization_error_ms <= config_.max_sync_tolerance_ms;

        pair.is_valid = left_quality_ok && right_quality_ok && sync_error_ok;

        if (!pair.is_valid) {
            statistics_.frames_dropped++;
            if (!sync_error_ok) {
                logWarning("Frame pair dropped due to sync error: " +
                          std::to_string(pair.synchronization_error_ms) + " ms");
            }
            return;
        }

        // Set camera info
        {
            std::lock_guard<std::mutex> lock(info_mutex_);
            if (left_info_received_) pair.left_camera_info = cached_left_info_;
            if (right_info_received_) pair.right_camera_info = cached_right_info_;
        }

        // Update statistics
        updateStatistics(pair);

        // Add to buffer if enabled
        if (config_.max_buffer_size > 0) {
            addToBuffer(pair);
        }

        // Call user callback
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (user_callback_) {
                user_callback_(pair);
            }
        }

        // Calculate processing latency
        auto end_time = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        // Update latency statistics
        statistics_.average_processing_latency_ms =
            (statistics_.average_processing_latency_ms * (statistics_.synchronized_pairs_created - 1) + latency_ms)
            / statistics_.synchronized_pairs_created;

    } catch (const std::exception& e) {
        statistics_.frames_dropped++;
        logError("Exception in processSynchronizedPair: " + std::string(e.what()));
    }
}

bool CameraSynchronizer::convertRosToOpenCV(const sensor_msgs::ImageConstPtr& msg, cv::Mat& output) {
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        output = cv_ptr->image;

        if (output.empty()) {
            logError("Converted image is empty");
            return false;
        }

        return true;

    } catch (cv_bridge::Exception& e) {
        logError("cv_bridge exception: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        logError("Exception in convertRosToOpenCV: " + std::string(e.what()));
        return false;
    }
}

bool CameraSynchronizer::validateFrameQuality(const cv::Mat& image) {
    if (image.empty()) {
        return false;
    }

    // Check image dimensions
    if (image.cols < 64 || image.rows < 64) {
        return false;
    }

    // Basic quality checks
    double sharpness = sync_utils::assessImageSharpness(image);
    double brightness = sync_utils::assessImageBrightness(image);
    bool has_motion_blur = sync_utils::detectMotionBlur(image);

    // Quality thresholds
    const double min_sharpness = config_.min_image_quality_threshold * 100.0;
    const double min_brightness = 0.1;
    const double max_brightness = 0.9;

    return (sharpness >= min_sharpness) &&
           (brightness >= min_brightness && brightness <= max_brightness) &&
           (!has_motion_blur);
}

double CameraSynchronizer::calculateSyncError(const ros::Time& t1, const ros::Time& t2) {
    return std::abs((t1 - t2).toSec()) * 1000.0; // Convert to milliseconds
}

void CameraSynchronizer::updateStatistics(const SynchronizedFramePair& pair) {
    statistics_.total_frames_received += 2; // Left + right
    statistics_.synchronized_pairs_created++;

    // Update sync error statistics
    double error_ms = pair.synchronization_error_ms;
    statistics_.average_sync_error_ms =
        (statistics_.average_sync_error_ms * (statistics_.synchronized_pairs_created - 1) + error_ms)
        / statistics_.synchronized_pairs_created;
    statistics_.max_sync_error_ms = std::max(statistics_.max_sync_error_ms, error_ms);

    // Calculate current FPS and success rate
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - statistics_.session_start).count() / 1000.0;

    if (elapsed_seconds > 0.0) {
        statistics_.current_fps = static_cast<double>(statistics_.synchronized_pairs_created) / elapsed_seconds;
    }

    uint64_t total_attempts = statistics_.synchronized_pairs_created + statistics_.frames_dropped;
    if (total_attempts > 0) {
        statistics_.sync_success_rate = static_cast<double>(statistics_.synchronized_pairs_created) / total_attempts;
    }

    // Adaptive synchronization if enabled
    if (config_.enable_adaptive_sync) {
        adaptSynchronizationParameters();
    }
}

void CameraSynchronizer::addToBuffer(const SynchronizedFramePair& pair) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    frame_buffer_.push_back(pair);

    // Prune buffer if necessary
    if (frame_buffer_.size() > static_cast<size_t>(config_.max_buffer_size)) {
        frame_buffer_.pop_front();
    }
}

void CameraSynchronizer::pruneBuffer() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    // Remove old frames based on timestamp
    auto current_time = ros::Time::now();
    double max_age_seconds = 1.0; // Remove frames older than 1 second

    frame_buffer_.erase(
        std::remove_if(frame_buffer_.begin(), frame_buffer_.end(),
            [&](const SynchronizedFramePair& pair) {
                return (current_time - pair.timestamp).toSec() > max_age_seconds;
            }),
        frame_buffer_.end());
}

void CameraSynchronizer::adaptSynchronizationParameters() {
    // Adaptive logic based on performance metrics
    if (statistics_.sync_success_rate < 0.8) {
        // Increase tolerance if success rate is low
        double new_tolerance = std::min(config_.max_sync_tolerance_ms * 1.2, 50.0);
        if (new_tolerance != config_.max_sync_tolerance_ms) {
            config_.max_sync_tolerance_ms = new_tolerance;
            logInfo("Adapted sync tolerance to " + std::to_string(new_tolerance) + " ms");
        }
    } else if (statistics_.sync_success_rate > 0.95 && statistics_.average_sync_error_ms < config_.max_sync_tolerance_ms * 0.5) {
        // Decrease tolerance if performance is very good
        double new_tolerance = std::max(config_.max_sync_tolerance_ms * 0.9, 1.0);
        if (new_tolerance != config_.max_sync_tolerance_ms) {
            config_.max_sync_tolerance_ms = new_tolerance;
            logInfo("Adapted sync tolerance to " + std::to_string(new_tolerance) + " ms");
        }
    }
}

void CameraSynchronizer::logError(const std::string& message) {
    ROS_ERROR("[CameraSynchronizer] %s", message.c_str());
}

void CameraSynchronizer::logWarning(const std::string& message) {
    ROS_WARN("[CameraSynchronizer] %s", message.c_str());
}

void CameraSynchronizer::logInfo(const std::string& message) {
    ROS_INFO("[CameraSynchronizer] %s", message.c_str());
}

// Factory function
std::unique_ptr<CameraSynchronizer> createCameraSynchronizer(const ros::NodeHandle& nh) {
    SynchronizationConfig config;

    // Load parameters from ROS parameter server
    nh.param<double>("sync/max_tolerance_ms", config.max_sync_tolerance_ms, 10.0);
    nh.param<double>("sync/target_fps", config.target_fps, 30.0);
    nh.param<int>("sync/max_buffer_size", config.max_buffer_size, 5);
    nh.param<bool>("sync/enable_frame_dropping", config.enable_frame_dropping, true);
    nh.param<double>("sync/min_quality_threshold", config.min_image_quality_threshold, 0.7);
    nh.param<bool>("sync/enable_adaptive_sync", config.enable_adaptive_sync, true);
    nh.param<bool>("sync/enable_zero_copy", config.enable_zero_copy, true);
    nh.param<bool>("sync/use_exact_time_sync", config.use_exact_time_sync, false);

    nh.param<std::string>("sync/left_image_topic", config.left_image_topic,
                         "/camera0/usb_cam_node/image_raw");
    nh.param<std::string>("sync/right_image_topic", config.right_image_topic,
                         "/camera1/usb_cam_node/image_raw");
    nh.param<std::string>("sync/left_camera_info_topic", config.left_camera_info_topic,
                         "/camera0/usb_cam_node/camera_info");
    nh.param<std::string>("sync/right_camera_info_topic", config.right_camera_info_topic,
                         "/camera1/usb_cam_node/camera_info");

    return std::make_unique<CameraSynchronizer>(config);
}

// Utility functions implementation
namespace sync_utils {

    double calculateTemporalOffset(const ros::Time& t1, const ros::Time& t2) {
        return (t1 - t2).toSec() * 1000.0; // Convert to milliseconds
    }

    bool validateCameraInfoConsistency(
        const sensor_msgs::CameraInfo& left_info,
        const sensor_msgs::CameraInfo& right_info) {

        // Check if image dimensions match
        if (left_info.width != right_info.width || left_info.height != right_info.height) {
            return false;
        }

        // Check if distortion models are similar
        if (left_info.distortion_model != right_info.distortion_model) {
            return false;
        }

        return true;
    }

    double assessImageSharpness(const cv::Mat& image) {
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

        return stddev.val[0] * stddev.val[0]; // Variance of Laplacian
    }

    double assessImageBrightness(const cv::Mat& image) {
        if (image.empty()) return 0.0;

        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }

        cv::Scalar mean_brightness = cv::mean(gray);
        return mean_brightness.val[0] / 255.0; // Normalize to [0, 1]
    }

    bool detectMotionBlur(const cv::Mat& image, double threshold) {
        if (image.empty()) return true;

        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }

        // Use variance of Laplacian as blur detection metric
        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_64F);
        cv::Scalar mean, stddev;
        cv::meanStdDev(laplacian, mean, stddev);

        double variance = stddev.val[0] * stddev.val[0];
        return variance < threshold; // Lower variance indicates more blur
    }
}

} // namespace camera_stitching