#ifndef LANE_DETECTION_CONFIG_HPP
#define LANE_DETECTION_CONFIG_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>

// Forward declarations
namespace lane_detection {
    struct YoloConfig;
    struct LaneSegmentationConfig;
}

namespace lane_detection {

/**
 * @brief Network model configuration matching your YAML specification
 */
struct NetworkConfig {
    struct ModelInfo {
        std::string name = "yolov8n-seg-lane";
        std::string type = "segmentation";
        std::string framework = "onnx";
    } model;

    struct InputConfig {
        std::string name = "images";
        std::vector<int> shape = {1, 3, 416, 416};
        std::string data_type = "FLOAT32";
        std::string color_format = "RGB";

        struct Normalization {
            std::vector<float> mean = {0.485f, 0.456f, 0.406f};
            std::vector<float> std = {0.229f, 0.224f, 0.225f};
            float scale = 255.0f;
        } normalization;
    } input;

    struct OutputConfig {
        struct Detection {
            std::string name = "output0";
            std::vector<int> shape = {1, 37, 3549};
            std::string data_type = "FLOAT32";
            std::string description = "Detection boxes and class scores";
        } detection;

        struct Segmentation {
            std::string name = "output1";
            std::vector<int> shape = {1, 32, 104, 104};
            std::string data_type = "FLOAT32";
            std::string description = "Segmentation prototype masks";
        } segmentation;
    } outputs;

    struct InferenceConfig {
        float confidence_threshold = 0.5f;
        float nms_threshold = 0.45f;
        int max_detections = 100;
        int target_fps = 30;
    } inference;

    struct OptimizationConfig {
        struct TensorRT {
            std::string precision = "FP16";
            int batch_size = 1;
            size_t workspace_size = 1073741824; // 1GB
        } tensorrt;
    } optimization;

    struct PostprocessingConfig {
        std::vector<int> lane_classes = {0};
        int min_area = 100;

        struct Morphology {
            int kernel_size = 3;
            int iterations = 2;
        } morphology;
    } postprocessing;

    struct ROSConfig {
        std::string input_topic = "/camera0/usb_cam_node/image_raw";

        struct OutputTopics {
            std::string lane_mask = "/camera0/lane_mask";
            std::string debug_image = "/camera0/lane_debug";
            std::string lane_detections = "/camera0/lane_detections";
        } output_topics;
    } ros;

    struct PerformanceConfig {
        bool enable_gpu_memory_pool = true;
        bool enable_zero_copy = true;
        bool async_inference = true;
        int max_queue_size = 3;
    } performance;
};

/**
 * @brief Configuration loader for YAML files
 */
class ConfigLoader {
public:
    /**
     * @brief Load network configuration from YAML file
     * @param config_path Path to YAML configuration file
     * @return Loaded network configuration
     */
    static NetworkConfig loadNetworkConfig(const std::string& config_path);

    /**
     * @brief Load configuration from ROS parameters
     * @param nh ROS NodeHandle for parameter access
     * @return Loaded network configuration
     */
    static NetworkConfig loadFromRosParams(const ros::NodeHandle& nh);

    /**
     * @brief Get default configuration
     * @return Default network configuration matching your specs
     */
    static NetworkConfig getDefaultConfig();

    /**
     * @brief Validate configuration parameters
     * @param config Configuration to validate
     * @return True if configuration is valid
     */
    static bool validateConfig(const NetworkConfig& config);

    /**
     * @brief Print configuration summary
     * @param config Configuration to print
     */
    static void printConfigSummary(const NetworkConfig& config);

private:
    static void loadModelConfig(const ros::NodeHandle& nh, NetworkConfig::ModelInfo& model);
    static void loadInputConfig(const ros::NodeHandle& nh, NetworkConfig::InputConfig& input);
    static void loadOutputConfig(const ros::NodeHandle& nh, NetworkConfig::OutputConfig& outputs);
    static void loadInferenceConfig(const ros::NodeHandle& nh, NetworkConfig::InferenceConfig& inference);
    static void loadOptimizationConfig(const ros::NodeHandle& nh, NetworkConfig::OptimizationConfig& optimization);
    static void loadPostprocessingConfig(const ros::NodeHandle& nh, NetworkConfig::PostprocessingConfig& postprocessing);
    static void loadROSConfig(const ros::NodeHandle& nh, NetworkConfig::ROSConfig& ros_config);
    static void loadPerformanceConfig(const ros::NodeHandle& nh, NetworkConfig::PerformanceConfig& performance);
};

/**
 * @brief Convert NetworkConfig to YoloConfig for detector initialization
 * @param network_config Source network configuration
 * @param model_path Path to the ONNX model file
 * @return YoloConfig suitable for YoloDetector initialization
 */
YoloConfig networkConfigToYoloConfig(const NetworkConfig& network_config,
                                    const std::string& model_path);

/**
 * @brief Convert NetworkConfig to LaneSegmentationConfig
 * @param network_config Source network configuration
 * @return LaneSegmentationConfig for processor initialization
 */
LaneSegmentationConfig networkConfigToSegmentationConfig(const NetworkConfig& network_config);

} // namespace lane_detection

#endif // LANE_DETECTION_CONFIG_HPP