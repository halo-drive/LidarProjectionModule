// Forward declarations to avoid circular dependencies
namespace lane_detection {
    struct YoloConfig;
    struct LaneDetection;
    class TensorBuffer;
    class PerformanceMonitor;
}#ifndef LANE_DETECTION_YOLO_DETECTOR_HPP
#define LANE_DETECTION_YOLO_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>
#include <NvInfer.h>
#include <cuda_runtime.h>

#include "tensor_utils.hpp"

namespace lane_detection {

/**
 * @brief Configuration structure for YOLOv8 lane detection
 */
struct YoloConfig {
    // Model paths
    std::string model_path;                    // Path to ONNX model
    std::string engine_path;                   // Path to TensorRT engine (optional)

    // Input configuration
    cv::Size input_size = cv::Size(416, 416);  // Model input dimensions
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};  // ImageNet normalization
    std::vector<float> std = {0.229f, 0.224f, 0.225f};   // ImageNet normalization
    float scale = 255.0f;                      // Pixel value scaling factor

    // Detection thresholds
    float confidence_threshold = 0.5f;         // Minimum confidence for detections
    float nms_threshold = 0.45f;               // Non-maximum suppression threshold
    int max_detections = 100;                  // Maximum number of detections

    // TensorRT optimization
    std::string precision = "FP16";            // Precision mode: FP32, FP16, INT8
    int batch_size = 1;                        // Batch size for inference
    size_t workspace_size = 1073741824;        // TensorRT workspace size (1GB)

    // Lane-specific configuration
    std::vector<int> lane_classes = {0};       // Class IDs for lane objects
    int min_area = 100;                        // Minimum area for valid lanes
    int morph_kernel_size = 3;                 // Morphological operation kernel size
    int morph_iterations = 2;                  // Number of morphological iterations
};

/**
 * @brief Lane detection result structure
 */
struct LaneDetection {
    cv::Rect bbox;                             // Bounding box in pixel coordinates
    float confidence;                          // Detection confidence score [0, 1]
    int class_id;                             // Class ID (0 for lane)
    cv::Mat mask;                             // Segmentation mask (if available)
    std::vector<cv::Point2f> contour_points;  // Lane contour points
};

/**
 * @brief YOLOv8 TensorRT-based lane detector
 */
class YoloDetector {
public:
    explicit YoloDetector(const YoloConfig& config);
    ~YoloDetector();

    // Non-copyable
    YoloDetector(const YoloDetector&) = delete;
    YoloDetector& operator=(const YoloDetector&) = delete;

    /**
     * @brief Initialize the detector with TensorRT engine
     * @return True on successful initialization, false otherwise
     */
    bool initialize();

    /**
     * @brief Detect lanes in the input image
     * @param image Input image (BGR format)
     * @param detections Output vector of lane detections
     * @return True on successful detection, false on error
     */
    bool detectLanes(const cv::Mat& image, std::vector<LaneDetection>& detections);

    /**
     * @brief Generate binary lane mask from detections
     * @param detections Lane detections from detectLanes()
     * @param original_size Size of the original image
     * @return Binary lane mask (CV_8UC1)
     */
    cv::Mat generateLaneMask(const std::vector<LaneDetection>& detections,
                            const cv::Size& original_size);

    /**
     * @brief Create debug visualization image
     * @param image Original input image
     * @param detections Lane detections to visualize
     * @return Image with overlaid detection results
     */
    cv::Mat drawDebugImage(const cv::Mat& image, const std::vector<LaneDetection>& detections);

    // Performance monitoring
    double getLastInferenceTime() const { return last_inference_time_; }
    size_t getGpuMemoryUsage() const { return gpu_memory_usage_; }

    // Configuration updates
    void updateConfig(const YoloConfig& new_config) { config_ = new_config; }
    const YoloConfig& getConfig() const { return config_; }

private:
    // Configuration
    YoloConfig config_;

    // TensorRT components
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // GPU memory buffers
    std::unique_ptr<TensorBuffer> input_buffer_;
    std::unique_ptr<TensorBuffer> detection_buffer_;
    std::unique_ptr<TensorBuffer> segmentation_buffer_;

    // CUDA stream for async operations
    cudaStream_t cuda_stream_;

    // Performance tracking
    double last_inference_time_;
    size_t gpu_memory_usage_;

    // Private methods
    /**
     * @brief Preprocess input image for model inference
     * @param image Input image in BGR format
     * @return Preprocessed image ready for inference
     */
    cv::Mat preprocessImage(const cv::Mat& image);

    /**
     * @brief Copy preprocessed image to GPU input buffer
     * @param image Preprocessed image
     * @return True on success, false on error
     */
    bool copyImageToGPU(const cv::Mat& image);

    /**
     * @brief Execute TensorRT inference
     * @return True on successful inference, false on error
     */
    bool runInference();

    /**
     * @brief Process raw model outputs into lane detections
     * @param original_size Size of the original input image
     * @return Vector of processed lane detections
     */
    std::vector<LaneDetection> processOutputs(const cv::Size& original_size);

    /**
     * @brief Process detection tensor output
     * @param detection_data Raw detection tensor data
     * @param segmentation_data Raw segmentation tensor data
     * @return Vector of lane detections
     */
    std::vector<LaneDetection> processDetections(const float* detection_data,
                                                const float* segmentation_data);

    /**
     * @brief Process segmentation mask for a specific detection
     * @param segmentation_data Raw segmentation tensor data
     * @param bbox Detection bounding box
     * @param original_size Original image size
     * @return Processed segmentation mask
     */
    cv::Mat processSegmentationMask(const float* segmentation_data,
                                   const cv::Rect& bbox,
                                   const cv::Size& original_size);

    /**
     * @brief Apply Non-Maximum Suppression to detections
     * @param boxes Vector of bounding boxes
     * @param scores Vector of confidence scores
     * @param nms_threshold NMS threshold
     * @return Indices of boxes to keep after NMS
     */
    std::vector<int> performNMS(const std::vector<cv::Rect>& boxes,
                               const std::vector<float>& scores,
                               float nms_threshold);

    /**
     * @brief Calculate Intersection over Union (IoU) between two boxes
     * @param box1 First bounding box
     * @param box2 Second bounding box
     * @return IoU value [0, 1]
     */
    float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);

    /**
     * @brief Apply morphological operations to clean up lane mask
     * @param mask Input binary mask
     * @return Cleaned binary mask
     */
    cv::Mat morphologyCleanup(const cv::Mat& mask);

    // Engine management
    bool buildEngine(const std::string& onnx_path);
    bool saveEngine(const std::string& engine_path);
    bool allocateBuffers();

    // Error handling
    bool checkCudaError(const std::string& operation);
    void logError(const std::string& message);
};

/**
 * @brief Factory function to create YoloDetector from configuration file
 * @param config_path Path to YAML configuration file
 * @return Unique pointer to YoloDetector instance
 */
std::unique_ptr<YoloDetector> createYoloDetector(const std::string& config_path);

} // namespace lane_detection

#endif // LANE_DETECTION_YOLO_DETECTOR_HPP