#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv2/opencv.hpp>
#include <memory>
#include <chrono>

#include "yolo_detector.hpp"
#include "lane_segmentation.hpp"
#include "tensor_utils.hpp"

class LaneDetectionNode {
public:
    LaneDetectionNode() : 
        nh_(""), 
        pnh_("~"),
        it_(nh_),
        initialized_(false),
        frame_count_(0),
        total_processing_time_(0.0)
    {
        initializeParameters();
        initializePublishersSubscribers();
        
        if (initializeDetector()) {
            initialized_ = true;
            ROS_INFO("Lane Detection Node initialized successfully");
        } else {
            ROS_ERROR("Failed to initialize Lane Detection Node");
        }
    }
    
    ~LaneDetectionNode() {
        if (performance_monitor_) {
            performance_monitor_->printStatistics();
        }
        ROS_INFO("Lane Detection Node shutdown complete");
    }
    
    bool isInitialized() const { return initialized_; }

private:
    // ROS components
    ros::NodeHandle nh_, pnh_;
    image_transport::ImageTransport it_;
    
    // Subscribers and Publishers
    image_transport::Subscriber image_sub_;
    image_transport::Publisher lane_mask_pub_;
    image_transport::Publisher debug_image_pub_;
    ros::Publisher lane_detections_pub_;
    ros::Publisher lane_markers_pub_;
    ros::Publisher statistics_pub_;
    
    // Core processing components
    std::unique_ptr<lane_detection::YoloDetector> yolo_detector_;
    std::unique_ptr<lane_detection::LaneSegmentationProcessor> lane_processor_;
    std::unique_ptr<lane_detection::PerformanceMonitor> performance_monitor_;
    
    // Configuration
    lane_detection::YoloConfig yolo_config_;
    lane_detection::LaneSegmentationConfig segmentation_config_;
    
    // Node parameters
    std::string input_topic_;
    std::string output_topic_prefix_;
    std::string frame_id_;
    bool enable_debug_output_;
    bool enable_performance_monitoring_;
    double target_fps_;
    
    // State tracking
    bool initialized_;
    int frame_count_;
    double total_processing_time_;
    ros::Time last_processing_time_;
    
    void initializeParameters() {
        // Input/Output topics
        pnh_.param<std::string>("input_topic", input_topic_, "/camera0/usb_cam_node/image_raw");
        pnh_.param<std::string>("output_topic_prefix", output_topic_prefix_, "/camera0/lane_");
        pnh_.param<std::string>("frame_id", frame_id_, "camera0_link");
        
        // Model configuration
        std::string config_path;
        pnh_.param<std::string>("config_path", config_path, 
                               "$(find LidarProjectionLane)/config/network_params/yolov8n_seg_lane.yaml");
        
        // Performance settings
        pnh_.param<bool>("enable_debug_output", enable_debug_output_, true);
        pnh_.param<bool>("enable_performance_monitoring", enable_performance_monitoring_, true);
        pnh_.param<double>("target_fps", target_fps_, 30.0);
        
        // Load YOLO configuration
        loadYoloConfig(config_path);
        loadSegmentationConfig();
        
        ROS_INFO("Lane Detection Node parameters loaded");
        ROS_INFO("Input topic: %s", input_topic_.c_str());
        ROS_INFO("Output prefix: %s", output_topic_prefix_.c_str());
        ROS_INFO("Target FPS: %.1f", target_fps_);
    }
    
    void initializePublishersSubscribers() {
        // Subscriber
        image_sub_ = it_.subscribe(input_topic_, 1, 
                                  &LaneDetectionNode::imageCallback, this);
        
        // Publishers
        lane_mask_pub_ = it_.advertise(output_topic_prefix_ + "mask", 1);
        debug_image_pub_ = it_.advertise(output_topic_prefix_ + "debug", 1);
        lane_detections_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
            output_topic_prefix_ + "detections", 1);
        lane_markers_pub_ = nh_.advertise<visualization_msgs::Marker>(
            output_topic_prefix_ + "markers", 1);
        
        if (enable_performance_monitoring_) {
            statistics_pub_ = nh_.advertise<std_msgs::Float64MultiArray>(
                output_topic_prefix_ + "statistics", 1);
        }
        
        ROS_INFO("Publishers and subscribers initialized");
    }
    
    bool initializeDetector() {
        try {
            // Initialize performance monitor
            if (enable_performance_monitoring_) {
                performance_monitor_ = std::make_unique<lane_detection::PerformanceMonitor>();
            }
            
            // Initialize YOLO detector
            yolo_detector_ = std::make_unique<lane_detection::YoloDetector>(yolo_config_);
            if (!yolo_detector_->initialize()) {
                ROS_ERROR("Failed to initialize YOLO detector");
                return false;
            }
            
            // Initialize lane segmentation processor
            lane_processor_ = std::make_unique<lane_detection::LaneSegmentationProcessor>(
                segmentation_config_);
            
            ROS_INFO("All components initialized successfully");
            return true;
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception during initialization: %s", e.what());
            return false;
        }
    }
    
    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        if (!initialized_) {
            ROS_WARN_THROTTLE(1.0, "Node not initialized, skipping frame");
            return;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Convert ROS image to OpenCV
            cv_bridge::CvImagePtr cv_ptr;
            try {
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            } catch (cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }
            
            // Process the image
            processFrame(cv_ptr->image, msg->header);
            
            // Performance monitoring
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();
            
            updatePerformanceStats(duration);
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in image callback: %s", e.what());
        }
    }
    
    void processFrame(const cv::Mat& image, const std_msgs::Header& header) {
        if (performance_monitor_) {
            performance_monitor_->startTiming("total_processing");
        }
        
        // Step 1: YOLO lane detection
        std::vector<lane_detection::LaneDetection> detections;
        if (performance_monitor_) {
            performance_monitor_->startTiming("yolo_inference");
        }
        
        bool detection_success = yolo_detector_->detectLanes(image, detections);
        
        if (performance_monitor_) {
            performance_monitor_->endTiming("yolo_inference");
        }
        
        if (!detection_success) {
            ROS_WARN("YOLO detection failed for current frame");
            return;
        }
        
        // Step 2: Generate lane mask
        cv::Mat lane_mask = yolo_detector_->generateLaneMask(detections, image.size());
        
        // Step 3: Process lane segmentation
        if (performance_monitor_) {
            performance_monitor_->startTiming("lane_processing");
        }
        
        lane_detection::LaneModel lane_model = lane_processor_->processLaneMask(
            lane_mask, image, header.stamp);
        lane_model.frame_id = header.frame_id.empty() ? frame_id_ : header.frame_id;
        
        if (performance_monitor_) {
            performance_monitor_->endTiming("lane_processing");
        }
        
        // Step 4: Publish results
        publishResults(image, lane_mask, lane_model, header);
        
        if (performance_monitor_) {
            performance_monitor_->endTiming("total_processing");
            
            // Record performance metrics
            performance_monitor_->recordInferenceTime(
                yolo_detector_->getLastInferenceTime());
            performance_monitor_->recordMemoryUsage(
                yolo_detector_->getGpuMemoryUsage());
        }
        
        frame_count_++;
    }
    
    void publishResults(const cv::Mat& image, 
                       const cv::Mat& lane_mask,
                       const lane_detection::LaneModel& lane_model,
                       const std_msgs::Header& header) {
        
        // Publish lane mask
        if (lane_mask_pub_.getNumSubscribers() > 0) {
            sensor_msgs::ImagePtr mask_msg = cv_bridge::CvImage(
                header, "mono8", lane_mask).toImageMsg();
            lane_mask_pub_.publish(mask_msg);
        }
        
        // Publish debug image
        if (enable_debug_output_ && debug_image_pub_.getNumSubscribers() > 0) {
            cv::Mat debug_image = lane_processor_->drawLaneModel(image, lane_model);
            sensor_msgs::ImagePtr debug_msg = cv_bridge::CvImage(
                header, "bgr8", debug_image).toImageMsg();
            debug_image_pub_.publish(debug_msg);
        }
        
        // Publish lane markers for RViz
        if (lane_markers_pub_.getNumSubscribers() > 0) {
            visualization_msgs::Marker marker = lane_processor_->laneModelToMarker(
                lane_model, lane_model.frame_id);
            marker.header = header;
            lane_markers_pub_.publish(marker);
        }
        
        // Publish performance statistics
        if (enable_performance_monitoring_ && statistics_pub_.getNumSubscribers() > 0) {
            publishPerformanceStats(header);
        }
    }
    
    void publishPerformanceStats(const std_msgs::Header& header) {
        std_msgs::Float64MultiArray stats_msg;
        stats_msg.layout.dim.resize(1);
        stats_msg.layout.dim[0].label = "statistics";
        stats_msg.layout.dim[0].size = 6;
        stats_msg.layout.dim[0].stride = 6;
        
        stats_msg.data.resize(6);
        stats_msg.data[0] = performance_monitor_->getCurrentFPS();
        stats_msg.data[1] = performance_monitor_->getAverageInferenceTime();
        stats_msg.data[2] = performance_monitor_->getMaxInferenceTime();
        stats_msg.data[3] = performance_monitor_->getMinInferenceTime();
        stats_msg.data[4] = static_cast<double>(performance_monitor_->getCurrentMemoryUsage()) / (1024 * 1024); // MB
        stats_msg.data[5] = static_cast<double>(frame_count_);
        
        statistics_pub_.publish(stats_msg);
    }
    
    void updatePerformanceStats(double processing_time_ms) {
        total_processing_time_ += processing_time_ms;
        last_processing_time_ = ros::Time::now();
        
        // Log performance every 100 frames
        if (frame_count_ % 100 == 0 && frame_count_ > 0) {
            double avg_processing_time = total_processing_time_ / frame_count_;
            double current_fps = 1000.0 / avg_processing_time;
            
            ROS_INFO("Performance Stats - Frame: %d, Avg Processing: %.2f ms, FPS: %.1f",
                    frame_count_, avg_processing_time, current_fps);
            
            if (performance_monitor_) {
                performance_monitor_->printStatistics();
            }
        }
    }
    
    void loadYoloConfig(const std::string& config_path) {
        // Load YOLO configuration from YAML file
        // This is a simplified version - in practice, you'd use yaml-cpp
        yolo_config_.model_path = ros::package::getPath("LidarProjectionLane") + 
                                 "/lane_detection/models/yolov8n-seg-lane.onnx";
        yolo_config_.engine_path = ros::package::getPath("LidarProjectionLane") + 
                                  "/lane_detection/models/yolov8n-seg-lane.engine";
        
        yolo_config_.input_size = cv::Size(416, 416);
        yolo_config_.confidence_threshold = 0.5f;
        yolo_config_.nms_threshold = 0.45f;
        yolo_config_.precision = "FP16";
        yolo_config_.batch_size = 1;
        yolo_config_.workspace_size = 1073741824; // 1GB
        
        ROS_INFO("YOLO configuration loaded");
    }
    
    void loadSegmentationConfig() {
        // Load segmentation configuration
        segmentation_config_.min_area_threshold = 100;
        segmentation_config_.max_area_threshold = 50000;
        segmentation_config_.contour_epsilon = 2.0;
        segmentation_config_.morph_kernel = cv::Size(3, 3);
        segmentation_config_.morph_iterations = 2;
        segmentation_config_.hough_threshold = 50;
        segmentation_config_.min_line_length = 30.0;
        segmentation_config_.max_line_gap = 10.0;
        segmentation_config_.enable_debug_output = enable_debug_output_;
        
        ROS_INFO("Lane segmentation configuration loaded");
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lane_detection_node");
    
    // Check CUDA availability
    lane_detection::utils::printCudaDeviceInfo();
    if (!lane_detection::utils::checkCudaCapability(7, 0)) {
        ROS_ERROR("CUDA capability 7.0 or higher required");
        return -1;
    }
    
    try {
        LaneDetectionNode node;
        
        if (!node.isInitialized()) {
            ROS_ERROR("Failed to initialize lane detection node");
            return -1;
        }
        
        ROS_INFO("Lane Detection Node starting...");
        ros::spin();
        
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in main: %s", e.what());
        return -1;
    }
    
    return 0;
}