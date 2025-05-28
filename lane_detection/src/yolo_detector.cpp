#include "yolo_detector.hpp"
#include "tensor_utils.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <ros/package.h>

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
    // This is a simplified implementation
    // In practice, you'd use the mask coefficients to generate the final mask
    cv::Mat mask = cv::Mat::zeros(104, 104, CV_32F); // Segmentation output size

    // For now, create a simple mask based on the bounding box
    cv::Rect scaled_bbox;
    scaled_bbox.x = bbox.x * 104 / original_size.width;
    scaled_bbox.y = bbox.y * 104 / original_size.height;
    scaled_bbox.width = bbox.width * 104 / original_size.width;
    scaled_bbox.height = bbox.height * 104 / original_size.height;

    mask(scaled_bbox) = 1.0f;

    // Convert to 8-bit
    cv::Mat mask_8u;
    mask.convertTo(mask_8u, CV_8UC1, 255.0);

    return mask_8u;
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

bool YoloDetector::buildEngine(const std::string& onnx_path) {
    engine_ = EngineUtils::buildEngineFromOnnx(onnx_path, config_.precision,
                                              config_.batch_size, config_.workspace_size);
    return engine_ != nullptr;
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