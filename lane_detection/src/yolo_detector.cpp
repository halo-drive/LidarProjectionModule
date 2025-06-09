#include "../include/yolo_detector.hpp"
#include "../include/tensor_utils.hpp"
#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <algorithm>
#include <chrono>


// External CUDA kernel declarations
extern "C" {
    cudaError_t launchPreprocessKernel(
        const uint8_t* input, float* output,
        int width, int height,
        const float* mean, const float* std, float scale,
        cudaStream_t stream);

    cudaError_t launchMaskProcessKernel(
        const float* input_mask, uint8_t* output_mask,
        int width, int height, float threshold,
        cudaStream_t stream);
}

namespace lane_detection {

YoloDetector::YoloDetector(const YoloConfig& config)
    : config_(config), last_inference_time_(0.0), gpu_memory_usage_(0) {

    // Create CUDA stream
    if (cudaStreamCreate(&cuda_stream_) != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream");
    }
}

YoloDetector::~YoloDetector() {
    if (cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
    }
}

bool YoloDetector::initialize() {
    try {
        // Create TensorRT runtime
        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(EngineUtils::logger_));
        if (!runtime_) {
            logError("Failed to create TensorRT runtime");
            return false;
        }

        // Try to load existing engine first
        std::string engine_path = config_.model_path;
        size_t pos = engine_path.find_last_of('.');
        if (pos != std::string::npos) {
            engine_path = engine_path.substr(0, pos) + ".engine";
        } else {
            engine_path += ".engine";
        }

        engine_ = EngineUtils::loadEngine(engine_path, *runtime_);

        // If engine loading fails, build from ONNX
        if (!engine_) {
            ROS_INFO("Engine not found, building from ONNX: %s", config_.model_path.c_str());
            engine_ = buildEngine(config_.model_path);
            if (!engine_) {
                logError("Failed to build engine from ONNX");
                return false;
            }

            // Save the built engine for future use
            if (!saveEngine(engine_path)) {
                ROS_WARN("Failed to save engine to %s", engine_path.c_str());
            }
        }

        // Create execution context
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
            engine_->createExecutionContext());
        if (!context_) {
            logError("Failed to create execution context");
            return false;
        }

        // Allocate GPU memory buffers
        if (!allocateBuffers()) {
            logError("Failed to allocate GPU buffers");
            return false;
        }

        ROS_INFO("YOLOv8 TensorRT detector initialized successfully");
        EngineUtils::printEngineInfo(*engine_);

        return true;

    } catch (const std::exception& e) {
        logError("Exception during initialization: " + std::string(e.what()));
        return false;
    }
}

bool YoloDetector::detectLanes(const cv::Mat& image, std::vector<LaneDetection>& detections) {
    if (!engine_ || !context_) {
        logError("Detector not initialized");
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        // Preprocess image
        cv::Mat processed_image = preprocessImage(image);

        // Copy preprocessed image to GPU input buffer
        if (!copyImageToGPU(processed_image)) {
            logError("Failed to copy image to GPU");
            return false;
        }

        // Run inference
        if (!runInference()) {
            logError("Inference failed");
            return false;
        }

        // Process outputs
        detections = processOutputs(image.size());

        // Calculate inference time
        auto end_time = std::chrono::high_resolution_clock::now();
        last_inference_time_ = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();

        return true;

    } catch (const std::exception& e) {
        logError("Exception during detection: " + std::string(e.what()));
        return false;
    }
}

cv::Mat YoloDetector::generateLaneMask(const std::vector<LaneDetection>& detections,
                                      const cv::Size& original_size) {
    cv::Mat lane_mask = cv::Mat::zeros(original_size, CV_8UC1);

    for (const auto& detection : detections) {
        if (std::find(config_.lane_classes.begin(), config_.lane_classes.end(),
                     detection.class_id) != config_.lane_classes.end()) {

            if (!detection.mask.empty()) {
                // Resize mask to original size
                cv::Mat resized_mask;
                cv::resize(detection.mask, resized_mask, original_size, 0, 0, cv::INTER_LINEAR);

                // Apply threshold
                cv::threshold(resized_mask, resized_mask, 127, 255, cv::THRESH_BINARY);

                // Add to lane mask
                cv::bitwise_or(lane_mask, resized_mask, lane_mask);
            }
        }
    }

    // Apply morphological operations for cleanup
    return morphologyCleanup(lane_mask);
}

cv::Mat YoloDetector::drawDebugImage(const cv::Mat& image,
                                    const std::vector<LaneDetection>& detections) {
    cv::Mat debug_image;
    image.copyTo(debug_image);

    for (const auto& detection : detections) {
        // Draw bounding box
        cv::rectangle(debug_image, detection.bbox, cv::Scalar(0, 255, 0), 2);

        // Draw confidence score
        std::string label = "Lane: " + std::to_string(detection.confidence);
        cv::putText(debug_image, label,
                   cv::Point(detection.bbox.x, detection.bbox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // Overlay mask if available
        if (!detection.mask.empty()) {
            cv::Mat colored_mask;
            cv::applyColorMap(detection.mask, colored_mask, cv::COLORMAP_JET);
            cv::addWeighted(debug_image, 0.7, colored_mask, 0.3, 0, debug_image);
        }
    }

    // Add performance info
    std::string perf_info = "Inference: " + std::to_string(last_inference_time_) + "ms";
    cv::putText(debug_image, perf_info, cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    return debug_image;
}

// Private methods implementation

cv::Mat YoloDetector::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;

    // Resize to model input size
    cv::resize(image, processed, config_.input_size, 0, 0, cv::INTER_LINEAR);

    // Convert BGR to RGB
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

    return processed;
}

bool YoloDetector::copyImageToGPU(const cv::Mat& image) {
    if (image.empty() || !input_buffer_) {
        return false;
    }

    // Get host pointer for input buffer
    uint8_t* host_input = static_cast<uint8_t*>(input_buffer_->getHostPtr());

    // Copy image data to host buffer
    std::memcpy(host_input, image.data, image.total() * image.elemSize());

    // Preprocess on GPU (BGR->RGB, normalize, HWC->CHW)
    float* gpu_input = static_cast<float*>(input_buffer_->getDevicePtr());

    // Device pointers for normalization parameters
    float* d_mean;
    float* d_std;
    cudaMalloc(&d_mean, 3 * sizeof(float));
    cudaMalloc(&d_std, 3 * sizeof(float));

    cudaMemcpy(d_mean, config_.mean.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_std, config_.std.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch preprocessing kernel
    cudaError_t result = launchPreprocessKernel(
        host_input, gpu_input,
        config_.input_size.width, config_.input_size.height,
        d_mean, d_std, config_.scale, cuda_stream_);

    // Cleanup
    cudaFree(d_mean);
    cudaFree(d_std);

    if (result != cudaSuccess) {
        logError("Preprocessing kernel failed: " + std::string(cudaGetErrorString(result)));
        return false;
    }

    return true;
}

bool YoloDetector::runInference() {
    // Set up bindings
    std::vector<void*> bindings(engine_->getNbBindings());

    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        if (engine_->bindingIsInput(i)) {
            bindings[i] = input_buffer_->getDevicePtr();
        } else {
            // Handle multiple outputs
            if (i == 1) { // Detection output
                bindings[i] = detection_buffer_->getDevicePtr();
            } else if (i == 2) { // Segmentation output
                bindings[i] = segmentation_buffer_->getDevicePtr();
            }
        }
    }

    // Execute the network
    bool success = context_->enqueueV2(bindings.data(), cuda_stream_, nullptr);

    if (!success) {
        logError("TensorRT inference execution failed");
        return false;
    }

    // Synchronize stream
    cudaStreamSynchronize(cuda_stream_);

    return checkCudaError("inference execution");
}

std::vector<LaneDetection> YoloDetector::processOutputs(const cv::Size& original_size) {
    std::vector<LaneDetection> detections;

    // Copy detection output from GPU to CPU
    std::vector<float> detection_data(37 * 3549); // Based on your network config
    detection_buffer_->copyToHost(detection_data.data(),
                                 detection_data.size() * sizeof(float), cuda_stream_);

    // Copy segmentation output from GPU to CPU
    std::vector<float> segmentation_data(32 * 104 * 104); // Based on your network config
    segmentation_buffer_->copyToHost(segmentation_data.data(),
                                    segmentation_data.size() * sizeof(float), cuda_stream_);

    // Process detections and segmentation masks
    detections = processDetections(detection_data.data(), segmentation_data.data());

    // Apply NMS
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> indices;

    for (const auto& det : detections) {
        boxes.push_back(det.bbox);
        scores.push_back(det.confidence);
    }

    std::vector<int> nms_indices = performNMS(boxes, scores, config_.nms_threshold);

    // Filter detections based on NMS results
    std::vector<LaneDetection> filtered_detections;
    for (int idx : nms_indices) {
        if (idx < detections.size()) {
            filtered_detections.push_back(detections[idx]);
        }
    }

    return filtered_detections;
}

std::vector<LaneDetection> YoloDetector::processDetections(const float* detection_data,
                                                          const float* segmentation_data) {
    std::vector<LaneDetection> detections;

    // YOLOv8 output format: [batch, 37, 3549]
    // 37 = 4 (bbox) + 1 (confidence) + 32 (mask coefficients)
    // 3549 = number of anchors

    const int num_anchors = 3549;
    const int detection_size = 37;

    for (int i = 0; i < num_anchors; ++i) {
        const float* anchor_data = detection_data + i * detection_size;

        // Extract bbox (center_x, center_y, width, height)
        float cx = anchor_data[0];
        float cy = anchor_data[1];
        float w = anchor_data[2];
        float h = anchor_data[3];

        // Extract confidence
        float confidence = anchor_data[4];

        // Filter by confidence threshold
        if (confidence < config_.confidence_threshold) {
            continue;
        }

        // Convert to corner coordinates
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        // Scale to input image size
        x1 *= config_.input_size.width;
        y1 *= config_.input_size.height;
        x2 *= config_.input_size.width;
        y2 *= config_.input_size.height;

        // Create detection
        LaneDetection detection;
        detection.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
        detection.confidence = confidence;
        detection.class_id = 0; // Lane class

        // Process segmentation mask
        const float* mask_coeffs = anchor_data + 5; // 32 coefficients
        detection.mask = processSegmentationMask(segmentation_data, detection.bbox,
                                               config_.input_size);

        if (!detection.mask.empty() &&
            cv::countNonZero(detection.mask) > config_.min_area) {
            detections.push_back(detection);
        }
    }

    return detections;
}

cv::Mat YoloDetector::processSegmentationMask(const float* segmentation_data,
                                             const cv::Rect& bbox,
                                             const cv::Size& original_size) {
    // Constants for segmentation output dimensions
    constexpr int SEG_OUTPUT_WIDTH = 104;
    constexpr int SEG_OUTPUT_HEIGHT = 104;
    constexpr float CONFIDENCE_THRESHOLD = 0.3f;

    // Initialize output mask
    cv::Mat mask = cv::Mat::zeros(SEG_OUTPUT_HEIGHT, SEG_OUTPUT_WIDTH, CV_32F);

    // Input validation
    if (!segmentation_data) {
        ROS_ERROR("[YoloDetector] Null segmentation data provided");
        return cv::Mat::zeros(SEG_OUTPUT_HEIGHT, SEG_OUTPUT_WIDTH, CV_8UC1);
    }

    if (bbox.area() <= 0) {
        ROS_WARN("[YoloDetector] Invalid bbox area: %d", bbox.area());
        return cv::Mat::zeros(SEG_OUTPUT_HEIGHT, SEG_OUTPUT_WIDTH, CV_8UC1);
    }

    if (original_size.width <= 0 || original_size.height <= 0) {
        ROS_ERROR("[YoloDetector] Invalid original image dimensions: %dx%d",
                  original_size.width, original_size.height);
        return cv::Mat::zeros(SEG_OUTPUT_HEIGHT, SEG_OUTPUT_WIDTH, CV_8UC1);
    }

    try {
        // Calculate scaling factors with bounds checking
        const float scale_x = static_cast<float>(SEG_OUTPUT_WIDTH) / original_size.width;
        const float scale_y = static_cast<float>(SEG_OUTPUT_HEIGHT) / original_size.height;

        // Scale bbox coordinates to segmentation output space
        cv::Rect scaled_bbox;
        scaled_bbox.x = static_cast<int>(std::round(bbox.x * scale_x));
        scaled_bbox.y = static_cast<int>(std::round(bbox.y * scale_y));
        scaled_bbox.width = static_cast<int>(std::round(bbox.width * scale_x));
        scaled_bbox.height = static_cast<int>(std::round(bbox.height * scale_y));

        // Critical bounds validation and clipping
        cv::Rect safe_bbox;
        safe_bbox.x = std::max(0, std::min(scaled_bbox.x, SEG_OUTPUT_WIDTH - 1));
        safe_bbox.y = std::max(0, std::min(scaled_bbox.y, SEG_OUTPUT_HEIGHT - 1));

        // Ensure bbox doesn't exceed image boundaries
        int max_width = SEG_OUTPUT_WIDTH - safe_bbox.x;
        int max_height = SEG_OUTPUT_HEIGHT - safe_bbox.y;

        safe_bbox.width = std::max(1, std::min(scaled_bbox.width, max_width));
        safe_bbox.height = std::max(1, std::min(scaled_bbox.height, max_height));

        // Final validation before ROI access
        if (safe_bbox.x >= 0 && safe_bbox.y >= 0 &&
            safe_bbox.x + safe_bbox.width <= SEG_OUTPUT_WIDTH &&
            safe_bbox.y + safe_bbox.height <= SEG_OUTPUT_HEIGHT &&
            safe_bbox.area() > 0) {

            // Method 1: Simple bbox-based mask (current implementation, made safe)
            cv::Rect roi_region = safe_bbox;
            mask(roi_region) = 1.0f;

            // Method 2: Advanced segmentation processing using mask coefficients
            // This provides more accurate lane boundaries
            if (config_.use_advanced_segmentation) {
                processAdvancedSegmentation(segmentation_data, mask, safe_bbox);
            }

        } else {
            ROS_WARN("[YoloDetector] Invalid ROI after bounds checking: (%d,%d,%d,%d) for %dx%d mask",
                     safe_bbox.x, safe_bbox.y, safe_bbox.width, safe_bbox.height,
                     SEG_OUTPUT_WIDTH, SEG_OUTPUT_HEIGHT);
        }

        // Apply morphological operations for cleanup
        if (cv::countNonZero(mask) > 0) {
            mask = applyMorphologicalCleanup(mask);
        }

        // Convert to 8-bit output with proper thresholding
        cv::Mat mask_8u;
        mask.convertTo(mask_8u, CV_8UC1, 255.0);
        cv::threshold(mask_8u, mask_8u, static_cast<int>(CONFIDENCE_THRESHOLD * 255),
                      255, cv::THRESH_BINARY);

        return mask_8u;

    } catch (const cv::Exception& e) {
        ROS_ERROR("[YoloDetector] OpenCV exception in processSegmentationMask: %s", e.what());
        return cv::Mat::zeros(SEG_OUTPUT_HEIGHT, SEG_OUTPUT_WIDTH, CV_8UC1);
    } catch (const std::exception& e) {
        ROS_ERROR("[YoloDetector] Exception in processSegmentationMask: %s", e.what());
        return cv::Mat::zeros(SEG_OUTPUT_HEIGHT, SEG_OUTPUT_WIDTH, CV_8UC1);
    }
}

// Additional helper function for advanced segmentation processing
void YoloDetector::processAdvancedSegmentation(const float* segmentation_data,
                                             cv::Mat& mask,
                                             const cv::Rect& roi) {
    // Advanced implementation using YOLOv8 mask coefficients
    // This processes the 32-channel segmentation output more accurately

    constexpr int NUM_MASK_COEFFS = 32;
    constexpr int SEG_MAP_SIZE = 104 * 104;

    try {
        // Extract mask coefficients for this detection
        // Note: In full implementation, mask coefficients would be passed from detection processing

        // For each pixel in the ROI, compute the final mask value
        for (int y = roi.y; y < roi.y + roi.height; ++y) {
            for (int x = roi.x; x < roi.x + roi.width; ++x) {
                if (x >= 0 && y >= 0 && x < 104 && y < 104) {
                    int pixel_idx = y * 104 + x;

                    // Simplified confidence calculation
                    // In full implementation, this would use mask coefficients and prototype masks
                    float confidence = segmentation_data[pixel_idx % SEG_MAP_SIZE];

                    if (confidence > 0.5f) {
                        mask.at<float>(y, x) = confidence;
                    }
                }
            }
        }

    } catch (const std::exception& e) {
        ROS_WARN("[YoloDetector] Exception in advanced segmentation: %s", e.what());
    }
}

// Enhanced morphological cleanup with parameter validation
cv::Mat YoloDetector::applyMorphologicalCleanup(const cv::Mat& input_mask) {
    if (input_mask.empty()) {
        return input_mask;
    }

    cv::Mat cleaned_mask;

    try {
        // Validate kernel size
        int kernel_size = std::max(1, std::min(config_.morph_kernel_size, 7));
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                                   cv::Size(kernel_size, kernel_size));

        // Apply closing operation to fill small gaps
        cv::morphologyEx(input_mask, cleaned_mask, cv::MORPH_CLOSE, kernel,
                        cv::Point(-1, -1), config_.morph_iterations);

        // Apply opening to remove small noise
        cv::morphologyEx(cleaned_mask, cleaned_mask, cv::MORPH_OPEN, kernel,
                        cv::Point(-1, -1), 1);

        return cleaned_mask;

    } catch (const cv::Exception& e) {
        ROS_WARN("[YoloDetector] Morphological cleanup failed: %s", e.what());
        return input_mask;
    }
}

std::vector<int> YoloDetector::performNMS(const std::vector<cv::Rect>& boxes,
                                         const std::vector<float>& scores,
                                         float nms_threshold) {
    std::vector<int> indices;
    std::vector<std::pair<float, int>> score_index_pairs;

    // Create score-index pairs
    for (int i = 0; i < scores.size(); ++i) {
        score_index_pairs.push_back({scores[i], i});
    }

    // Sort by score (descending)
    std::sort(score_index_pairs.rbegin(), score_index_pairs.rend());

    std::vector<bool> suppressed(boxes.size(), false);

    for (const auto& pair : score_index_pairs) {
        int idx = pair.second;
        if (suppressed[idx]) continue;

        indices.push_back(idx);

        // Suppress overlapping boxes
        for (int i = 0; i < boxes.size(); ++i) {
            if (i == idx || suppressed[i]) continue;

            float iou = calculateIoU(boxes[idx], boxes[i]);
            if (iou > nms_threshold) {
                suppressed[i] = true;
            }
        }
    }

    return indices;
}

float YoloDetector::calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    cv::Rect intersection = box1 & box2;
    float inter_area = intersection.area();

    float union_area = box1.area() + box2.area() - inter_area;

    return (union_area > 0) ? (inter_area / union_area) : 0.0f;
}

cv::Mat YoloDetector::morphologyCleanup(const cv::Mat& mask) {
    cv::Mat cleaned;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                              cv::Size(config_.morph_kernel_size,
                                                      config_.morph_kernel_size));

    cv::morphologyEx(mask, cleaned, cv::MORPH_CLOSE, kernel,
                    cv::Point(-1, -1), config_.morph_iterations);

    return cleaned;
}

std::unique_ptr<nvinfer1::ICudaEngine> YoloDetector::buildEngine(const std::string& onnx_path) {
    return EngineUtils::buildEngineFromOnnx(onnx_path, config_.precision,
                                           config_.batch_size, config_.workspace_size);
}

bool YoloDetector::saveEngine(const std::string& engine_path) {
    if (!engine_) return false;
    return EngineUtils::saveEngine(*engine_, engine_path);
}

bool YoloDetector::allocateBuffers() {
    try {
        // Calculate buffer sizes based on engine bindings
        for (int i = 0; i < engine_->getNbBindings(); ++i) {
            auto dims = engine_->getBindingDimensions(i);
            auto dtype = engine_->getBindingDataType(i);
            size_t size = utils::calculateTensorSize(dims, dtype);

            if (engine_->bindingIsInput(i)) {
                input_buffer_ = std::make_unique<TensorBuffer>(size, dtype);
                ROS_INFO("Allocated input buffer: %zu bytes", size);
            } else {
                // Handle multiple outputs
                if (i == 1) { // Detection output
                    detection_buffer_ = std::make_unique<TensorBuffer>(size, dtype);
                    ROS_INFO("Allocated detection buffer: %zu bytes", size);
                } else if (i == 2) { // Segmentation output
                    segmentation_buffer_ = std::make_unique<TensorBuffer>(size, dtype);
                    ROS_INFO("Allocated segmentation buffer: %zu bytes", size);
                }
            }

            gpu_memory_usage_ += size;
        }

        ROS_INFO("Total GPU memory allocated: %zu MB", gpu_memory_usage_ / (1024 * 1024));
        return true;

    } catch (const std::exception& e) {
        logError("Failed to allocate buffers: " + std::string(e.what()));
        return false;
    }
}

bool YoloDetector::checkCudaError(const std::string& operation) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        logError("CUDA error in " + operation + ": " + std::string(cudaGetErrorString(error)));
        return false;
    }
    return true;
}

void YoloDetector::logError(const std::string& message) {
    ROS_ERROR("[YoloDetector] %s", message.c_str());
}

// Factory function
std::unique_ptr<YoloDetector> createYoloDetector(const std::string& config_path) {
    // Load configuration from YAML file
    YoloConfig config;

    // Set default paths relative to package
    std::string package_path = ros::package::getPath("lane_fusion");
    config.model_path = package_path + "/lane_detection/models/yolov8n-seg-lane.onnx";
    config.engine_path = package_path + "/lane_detection/models/yolov8n-seg-lane.engine";

    // Use your network parameters
    config.input_size = cv::Size(416, 416);
    config.mean = {0.485f, 0.456f, 0.406f};
    config.std = {0.229f, 0.224f, 0.225f};
    config.scale = 255.0f;
    config.confidence_threshold = 0.5f;
    config.nms_threshold = 0.45f;
    config.max_detections = 100;
    config.precision = "FP16";
    config.batch_size = 1;
    config.workspace_size = 1073741824; // 1GB
    config.lane_classes = {0};
    config.min_area = 100;
    config.morph_kernel_size = 3;
    config.morph_iterations = 2;

    return std::make_unique<YoloDetector>(config);
}

} // namespace lane_detection