#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Header.h>
#include <visualization_msgs/Marker.h>

#include <opencv2/opencv.hpp>
#include <memory>
#include <chrono>
#include <atomic>
#include <thread>

#include "camera_synchronizer.hpp"
#include "panoramic_stitcher.hpp"
#include "tensor_utils.hpp"

class CameraStitchingNode {
public:
    CameraStitchingNode() :
        nh_(""),
        pnh_("~"),
        it_(nh_),
        initialized_(false),
        running_(false),
        frame_count_(0),
        total_processing_time_(0.0)
    {
        initializeParameters();
        initializePublishersSubscribers();

        if (initializeComponents()) {
            initialized_ = true;
            ROS_INFO("Camera Stitching Node initialized successfully");
        } else {
            ROS_ERROR("Failed to initialize Camera Stitching Node");
        }
    }

    ~CameraStitchingNode() {
        shutdown();
        ROS_INFO("Camera Stitching Node shutdown complete");
    }

    bool start() {
        if (!initialized_) {
            ROS_ERROR("Cannot start: node not initialized");
            return false;
        }

        // Start camera synchronizer
        if (!synchronizer_->start()) {
            ROS_ERROR("Failed to start camera synchronizer");
            return false;
        }

        running_ = true;
        ROS_INFO("Camera Stitching Node started successfully");

        // Start monitoring thread for system health
        std::thread monitoring_thread(&CameraStitchingNode::monitoringLoop, this);
        monitoring_thread.detach();

        return true;
    }

    bool isInitialized() const { return initialized_; }

private:
    // ROS components
    ros::NodeHandle nh_, pnh_;
    image_transport::ImageTransport it_;

    // Publishers
    image_transport::Publisher panorama_pub_;
    image_transport::Publisher debug_pub_;
    ros::Publisher statistics_pub_;
    ros::Publisher performance_pub_;

    // Core processing components
    std::unique_ptr<camera_stitching::CameraSynchronizer> synchronizer_;
    std::unique_ptr<camera_stitching::PanoramicStitcher> stitcher_;
    std::unique_ptr<lane_detection::PerformanceMonitor> performance_monitor_;

    // Node parameters
    std::string output_panorama_topic_;
    std::string output_debug_topic_;
    std::string output_statistics_topic_;
    std::string frame_id_;
    bool enable_debug_output_;
    bool enable_performance_monitoring_;
    bool enable_real_time_mode_;
    double target_fps_;
    double max_processing_latency_ms_;

    // Processing control
    std::atomic<bool> initialized_;
    std::atomic<bool> running_;
    std::atomic<bool> should_shutdown_;

    // Performance tracking
    int frame_count_;
    double total_processing_time_;
    ros::Time last_processing_time_;

    // Statistics
    struct ProcessingStatistics {
        uint64_t total_pairs_received = 0;
        uint64_t successful_stitches = 0;
        uint64_t failed_stitches = 0;
        uint64_t frames_skipped = 0;

        double average_processing_time_ms = 0.0;
        double max_processing_time_ms = 0.0;
        double current_fps = 0.0;

        double average_stitch_quality = 0.0;
        double average_overlap_percentage = 0.0;

        std::chrono::time_point<std::chrono::high_resolution_clock> session_start;
    } processing_stats_;

    void initializeParameters() {
        // Output topics
        pnh_.param<std::string>("output_panorama_topic", output_panorama_topic_,
                               "/camera_stitching/panorama");
        pnh_.param<std::string>("output_debug_topic", output_debug_topic_,
                               "/camera_stitching/debug");
        pnh_.param<std::string>("output_statistics_topic", output_statistics_topic_,
                               "/camera_stitching/statistics");
        pnh_.param<std::string>("frame_id", frame_id_, "panorama_link");

        // Processing settings
        pnh_.param<bool>("enable_debug_output", enable_debug_output_, true);
        pnh_.param<bool>("enable_performance_monitoring", enable_performance_monitoring_, true);
        pnh_.param<bool>("enable_real_time_mode", enable_real_time_mode_, true);
        pnh_.param<double>("target_fps", target_fps_, 15.0);
        pnh_.param<double>("max_processing_latency_ms", max_processing_latency_ms_, 100.0);

        processing_stats_.session_start = std::chrono::high_resolution_clock::now();
        should_shutdown_ = false;

        ROS_INFO("Camera Stitching Node parameters loaded");
        ROS_INFO("Output panorama topic: %s", output_panorama_topic_.c_str());
        ROS_INFO("Debug output: %s", enable_debug_output_ ? "enabled" : "disabled");
        ROS_INFO("Target FPS: %.1f", target_fps_);
        ROS_INFO("Real-time mode: %s", enable_real_time_mode_ ? "enabled" : "disabled");
    }

    void initializePublishersSubscribers() {
        // Publishers
        panorama_pub_ = it_.advertise(output_panorama_topic_, 1);

        if (enable_debug_output_) {
            debug_pub_ = it_.advertise(output_debug_topic_, 1);
        }

        if (enable_performance_monitoring_) {
            statistics_pub_ = nh_.advertise<std_msgs::Float64MultiArray>(
                output_statistics_topic_, 1);
            performance_pub_ = nh_.advertise<std_msgs::Float64MultiArray>(
                "/camera_stitching/performance", 1);
        }

        ROS_INFO("Publishers and subscribers initialized");
    }

    bool initializeComponents() {
        try {
            // Initialize performance monitor
            if (enable_performance_monitoring_) {
                performance_monitor_ = std::make_unique<lane_detection::PerformanceMonitor>();
            }

            // Initialize camera synchronizer
            synchronizer_ = camera_stitching::createCameraSynchronizer(pnh_);
            if (!synchronizer_->initialize()) {
                ROS_ERROR("Failed to initialize camera synchronizer");
                return false;
            }

            // Register callback for synchronized frame pairs
            synchronizer_->registerFramePairCallback(
                std::bind(&CameraStitchingNode::handleSynchronizedFrames, this, std::placeholders::_1));

            // Initialize panoramic stitcher
            stitcher_ = camera_stitching::createPanoramicStitcher(pnh_);
            if (!stitcher_->initialize()) {
                ROS_ERROR("Failed to initialize panoramic stitcher");
                return false;
            }

            ROS_INFO("All components initialized successfully");
            return true;

        } catch (const std::exception& e) {
            ROS_ERROR("Exception during component initialization: %s", e.what());
            return false;
        }
    }

    void handleSynchronizedFrames(const camera_stitching::SynchronizedFramePair& frame_pair) {
        if (!running_ || should_shutdown_) {
            return;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        if (performance_monitor_) {
            performance_monitor_->startTiming("total_processing");
        }

        try {
            processing_stats_.total_pairs_received++;

            // Check frame quality and validity
            if (!frame_pair.is_valid) {
                processing_stats_.frames_skipped++;
                ROS_WARN_THROTTLE(5.0, "Skipped invalid frame pair");
                return;
            }

            // Real-time mode: Skip frame if processing is falling behind
            if (enable_real_time_mode_) {
                ros::Time current_ros_time = ros::Time::now();
                double time_since_last_ms = (current_ros_time - last_processing_time_).toSec() * 1000.0;

                double expected_interval_ms = 1000.0 / target_fps_;
                if (time_since_last_ms < expected_interval_ms * 0.8) {
                    processing_stats_.frames_skipped++;
                    return; // Skip this frame to maintain real-time performance
                }
            }

            // Process frame pair through stitching pipeline
            camera_stitching::PanoramicResult result;

            if (performance_monitor_) {
                performance_monitor_->startTiming("stitching");
            }

            bool stitch_success = stitcher_->stitch(frame_pair, result);

            if (performance_monitor_) {
                performance_monitor_->endTiming("stitching");
            }

            if (stitch_success && result.is_valid) {
                processing_stats_.successful_stitches++;

                // Publish results
                publishResults(result, frame_pair);

                // Update performance statistics
                updatePerformanceStatistics(result);

            } else {
                processing_stats_.failed_stitches++;
                ROS_WARN_THROTTLE(10.0, "Stitching failed for frame pair");
            }

            // Calculate processing time
            auto end_time = std::chrono::high_resolution_clock::now();
            double processing_time_ms = std::chrono::duration<double, std::milli>(
                end_time - start_time).count();

            // Update processing statistics
            total_processing_time_ += processing_time_ms;
            frame_count_++;
            last_processing_time_ = ros::Time::now();

            if (processing_stats_.successful_stitches > 0) {
                processing_stats_.average_processing_time_ms =
                    (processing_stats_.average_processing_time_ms * (processing_stats_.successful_stitches - 1)
                     + processing_time_ms) / processing_stats_.successful_stitches;
            }

            processing_stats_.max_processing_time_ms =
                std::max(processing_stats_.max_processing_time_ms, processing_time_ms);

            // Calculate current FPS using session elapsed time
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - processing_stats_.session_start).count() / 1000.0;

            if (elapsed_seconds > 0.0) {
                processing_stats_.current_fps = static_cast<double>(processing_stats_.successful_stitches) / elapsed_seconds;
            }

            // Performance monitoring
            if (performance_monitor_) {
                performance_monitor_->endTiming("total_processing");
                performance_monitor_->recordInferenceTime(processing_time_ms);
            }

            // Publish performance statistics periodically
            if (frame_count_ % 30 == 0) { // Every 30 frames
                publishPerformanceStatistics();

                // Log performance every 100 frames
                if (frame_count_ % 100 == 0) {
                    logPerformanceStatistics();
                }
            }

            // Check for performance issues
            if (processing_time_ms > max_processing_latency_ms_) {
                ROS_WARN_THROTTLE(30.0, "Processing latency exceeded threshold: %.2f ms > %.2f ms",
                                 processing_time_ms, max_processing_latency_ms_);
            }

        } catch (const std::exception& e) {
            processing_stats_.failed_stitches++;
            ROS_ERROR("Exception in frame processing: %s", e.what());

            if (performance_monitor_) {
                performance_monitor_->endTiming("total_processing");
            }
        }
}

    void publishResults(const camera_stitching::PanoramicResult& result,
                       const camera_stitching::SynchronizedFramePair& frame_pair) {

        // Publish panoramic image
        if (panorama_pub_.getNumSubscribers() > 0) {
            std_msgs::Header header;
            header.stamp = frame_pair.timestamp;
            header.frame_id = frame_id_;

            sensor_msgs::ImagePtr panorama_msg = cv_bridge::CvImage(
                header, sensor_msgs::image_encodings::BGR8, result.panoramic_image).toImageMsg();
            panorama_pub_.publish(panorama_msg);
        }

        // Publish debug visualization if enabled
        if (enable_debug_output_ && debug_pub_.getNumSubscribers() > 0) {
            publishDebugVisualization(result, frame_pair);
        }
    }

    void publishDebugVisualization(const camera_stitching::PanoramicResult& result,
                                  const camera_stitching::SynchronizedFramePair& frame_pair) {
        try {
            // Create debug image with overlay information
            cv::Mat debug_image = result.panoramic_image.clone();

            // Draw ROI rectangles
            cv::rectangle(debug_image, result.left_roi, cv::Scalar(0, 255, 0), 2);
            cv::rectangle(debug_image, result.right_roi, cv::Scalar(0, 0, 255), 2);

            // Draw overlap region
            cv::Rect overlap = result.left_roi & result.right_roi;
            if (overlap.area() > 0) {
                cv::rectangle(debug_image, overlap, cv::Scalar(255, 255, 0), 2);
            }

            // Add text information
            std::vector<std::string> info_lines = {
                "Quality: " + std::to_string(result.stitching_quality),
                "Overlap: " + std::to_string(result.overlap_percentage) + "%",
                "Process Time: " + std::to_string(result.processing_time_ms) + "ms",
                "Sync Error: " + std::to_string(frame_pair.synchronization_error_ms) + "ms",
                "FPS: " + std::to_string(processing_stats_.current_fps),
                "Size: " + std::to_string(result.output_size.width) + "x" + std::to_string(result.output_size.height)
            };

            int y_offset = 30;
            for (const auto& line : info_lines) {
                cv::putText(debug_image, line, cv::Point(10, y_offset),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
                cv::putText(debug_image, line, cv::Point(10, y_offset),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
                y_offset += 25;
            }

            // Publish debug image
            std_msgs::Header header;
            header.stamp = frame_pair.timestamp;
            header.frame_id = frame_id_;

            sensor_msgs::ImagePtr debug_msg = cv_bridge::CvImage(
                header, sensor_msgs::image_encodings::BGR8, debug_image).toImageMsg();
            debug_pub_.publish(debug_msg);

        } catch (const std::exception& e) {
            ROS_ERROR("Exception creating debug visualization: %s", e.what());
        }
    }

    void publishPerformanceStatistics() {
        if (!enable_performance_monitoring_ || statistics_pub_.getNumSubscribers() == 0) {
            return;
        }

        std_msgs::Float64MultiArray stats_msg;
        stats_msg.layout.dim.resize(1);
        stats_msg.layout.dim[0].label = "statistics";
        stats_msg.layout.dim[0].size = 12;
        stats_msg.layout.dim[0].stride = 12;

        stats_msg.data.resize(12);
        stats_msg.data[0] = static_cast<double>(processing_stats_.total_pairs_received);
        stats_msg.data[1] = static_cast<double>(processing_stats_.successful_stitches);
        stats_msg.data[2] = static_cast<double>(processing_stats_.failed_stitches);
        stats_msg.data[3] = static_cast<double>(processing_stats_.frames_skipped);
        stats_msg.data[4] = processing_stats_.average_processing_time_ms;
        stats_msg.data[5] = processing_stats_.max_processing_time_ms;
        stats_msg.data[6] = processing_stats_.current_fps;
        stats_msg.data[7] = processing_stats_.average_stitch_quality;
        stats_msg.data[8] = processing_stats_.average_overlap_percentage;
        stats_msg.data[9] = stitcher_->getLastStitchQuality();
        stats_msg.data[10] = synchronizer_->getCurrentSyncQuality();
        stats_msg.data[11] = synchronizer_->getAverageFrameRate();

        statistics_pub_.publish(stats_msg);

        // Publish detailed performance data if available
        if (performance_monitor_ && performance_pub_.getNumSubscribers() > 0) {
            std_msgs::Float64MultiArray perf_msg;
            perf_msg.layout.dim.resize(1);
            perf_msg.layout.dim[0].label = "performance";
            perf_msg.layout.dim[0].size = 6;
            perf_msg.layout.dim[0].stride = 6;

            perf_msg.data.resize(6);
            perf_msg.data[0] = performance_monitor_->getCurrentFPS();
            perf_msg.data[1] = performance_monitor_->getAverageInferenceTime();
            perf_msg.data[2] = performance_monitor_->getMaxInferenceTime();
            perf_msg.data[3] = performance_monitor_->getMinInferenceTime();
            perf_msg.data[4] = static_cast<double>(performance_monitor_->getCurrentMemoryUsage()) / (1024 * 1024);
            perf_msg.data[5] = static_cast<double>(frame_count_);

            performance_pub_.publish(perf_msg);
        }
    }

    void updatePerformanceStatistics(const camera_stitching::PanoramicResult& result) {
        // Update quality statistics
        if (processing_stats_.successful_stitches > 0) {
            processing_stats_.average_stitch_quality =
                (processing_stats_.average_stitch_quality * (processing_stats_.successful_stitches - 1)
                 + result.stitching_quality) / processing_stats_.successful_stitches;

            processing_stats_.average_overlap_percentage =
                (processing_stats_.average_overlap_percentage * (processing_stats_.successful_stitches - 1)
                 + result.overlap_percentage) / processing_stats_.successful_stitches;
        }
    }

    void logPerformanceStatistics() {
        ROS_INFO("=== Camera Stitching Performance ===");
        ROS_INFO("Frame: %d, Success Rate: %.1f%%, FPS: %.2f",
                frame_count_,
                (static_cast<double>(processing_stats_.successful_stitches) /
                 std::max(1UL, processing_stats_.total_pairs_received)) * 100.0,
                processing_stats_.current_fps);
        ROS_INFO("Avg Processing: %.2f ms, Max: %.2f ms",
                processing_stats_.average_processing_time_ms,
                processing_stats_.max_processing_time_ms);
        ROS_INFO("Avg Quality: %.3f, Avg Overlap: %.1f%%",
                processing_stats_.average_stitch_quality,
                processing_stats_.average_overlap_percentage);
        ROS_INFO("Sync Quality: %.3f, Sync FPS: %.2f",
                synchronizer_->getCurrentSyncQuality(),
                synchronizer_->getAverageFrameRate());
        ROS_INFO("===================================");

        // Print detailed component statistics
        if (frame_count_ % 500 == 0) {
            synchronizer_->printStatistics();
            stitcher_->printStatistics();
            if (performance_monitor_) {
                performance_monitor_->printStatistics();
            }
        }
    }



    void stop() {
        if (!running_) return;

        running_ = false;

        if (synchronizer_) {
            synchronizer_->stop();
        }

        ROS_INFO("Camera Stitching Node stopped");
    }

    void shutdown() {
        should_shutdown_ = true;
        stop();

        // Final statistics
        if (frame_count_ > 0) {
            ROS_INFO("=== Final Camera Stitching Statistics ===");
            ROS_INFO("Total frames processed: %d", frame_count_);
            ROS_INFO("Average processing time: %.2f ms", total_processing_time_ / frame_count_);
            ROS_INFO("Successful stitches: %lu", processing_stats_.successful_stitches);
            ROS_INFO("Failed stitches: %lu", processing_stats_.failed_stitches);
            ROS_INFO("Final success rate: %.1f%%",
                    (static_cast<double>(processing_stats_.successful_stitches) /
                     std::max(1UL, processing_stats_.total_pairs_received)) * 100.0);
            ROS_INFO("=====================================");
        }
    }

    void monitoringLoop() {
        ros::Rate monitor_rate(1.0); // 1 Hz monitoring

        while (running_ && !should_shutdown_ && ros::ok()) {
            try {
                // Monitor system health
                checkSystemHealth();

                // Adaptive performance adjustments
                if (enable_real_time_mode_) {
                    adaptivePerformanceControl();
                }

                monitor_rate.sleep();

            } catch (const std::exception& e) {
                ROS_ERROR("Exception in monitoring loop: %s", e.what());
            }
        }
    }

    void checkSystemHealth() {
        // Check processing performance
        if (processing_stats_.current_fps < target_fps_ * 0.7) {
            ROS_WARN_THROTTLE(30.0, "Processing FPS below target: %.2f < %.2f",
                             processing_stats_.current_fps, target_fps_);
        }

        // Check synchronization health
        double sync_quality = synchronizer_->getCurrentSyncQuality();
        if (sync_quality < 0.8) {
            ROS_WARN_THROTTLE(30.0, "Synchronization quality degraded: %.3f", sync_quality);
        }

        // Check memory usage if performance monitor is available
        if (performance_monitor_) {
            size_t memory_usage_mb = performance_monitor_->getCurrentMemoryUsage() / (1024 * 1024);
            if (memory_usage_mb > 1024) { // 1GB threshold
                ROS_WARN_THROTTLE(60.0, "High memory usage detected: %zu MB", memory_usage_mb);
            }
        }
    }

    void adaptivePerformanceControl() {
        // Adaptive quality control based on performance
        if (processing_stats_.average_processing_time_ms > max_processing_latency_ms_) {
            // Could implement adaptive quality reduction here
            ROS_DEBUG_THROTTLE(60.0, "Considering performance optimization due to high latency");
        }

        // Adaptive frame skipping
        if (processing_stats_.current_fps < target_fps_ * 0.5) {
            // More aggressive frame skipping
            ROS_DEBUG_THROTTLE(60.0, "Implementing adaptive frame skipping due to low FPS");
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "camera_stitching_node");

    // Check CUDA availability
    lane_detection::utils::printCudaDeviceInfo();
    if (!lane_detection::utils::checkCudaCapability(7, 0)) {
        ROS_WARN("CUDA capability 7.0 or higher recommended for optimal performance");
    }

    try {
        CameraStitchingNode node;

        if (!node.isInitialized()) {
            ROS_ERROR("Failed to initialize camera stitching node");
            return -1;
        }

        // Start the node
        if (!node.start()) {
            ROS_ERROR("Failed to start camera stitching node");
            return -1;
        }

        ROS_INFO("Camera Stitching Node running...");
        ros::spin();

    } catch (const std::exception& e) {
        ROS_ERROR("Exception in main: %s", e.what());
        return -1;
    }

    ROS_INFO("Camera Stitching Node terminated");
    return 0;
}