#ifndef LANE_DETECTION_TENSOR_UTILS_HPP
#define LANE_DETECTION_TENSOR_UTILS_HPP

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <mutex>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

namespace lane_detection {

/**
 * @brief TensorRT Logger implementation
 */
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;

private:
    const char* getSeverityString(Severity severity);
};

/**
 * @brief CUDA tensor buffer management with automatic memory allocation
 */
class TensorBuffer {
public:
    explicit TensorBuffer(size_t size, nvinfer1::DataType data_type = nvinfer1::DataType::kFLOAT);
    ~TensorBuffer();

    // Move semantics only (no copy)
    TensorBuffer(const TensorBuffer&) = delete;
    TensorBuffer& operator=(const TensorBuffer&) = delete;
    TensorBuffer(TensorBuffer&& other) noexcept;
    TensorBuffer& operator=(TensorBuffer&& other) noexcept;

    // Memory access
    void* getHostPtr() const { return host_ptr_; }
    void* getDevicePtr() const { return device_ptr_; }
    size_t getSize() const { return size_; }
    nvinfer1::DataType getDataType() const { return data_type_; }

    // Memory operations
    bool hostToDevice(cudaStream_t stream = 0);
    bool deviceToHost(cudaStream_t stream = 0);
    bool copyFromHost(const void* src, size_t size, cudaStream_t stream = 0);
    bool copyToHost(void* dst, size_t size, cudaStream_t stream = 0);
    bool zeroInitialize(cudaStream_t stream = 0);

private:
    void* device_ptr_;
    void* host_ptr_;
    size_t size_;
    nvinfer1::DataType data_type_;

    bool allocateMemory();
    void deallocateMemory();
};

/**
 * @brief TensorRT Engine utilities for building and loading engines
 */
class EngineUtils {
public:
    static TensorRTLogger logger_;

    /**
     * @brief Build TensorRT engine from ONNX model
     * @param onnx_path Path to ONNX model file
     * @param precision Precision mode ("FP16", "FP32", "INT8")
     * @param batch_size Maximum batch size
     * @param workspace_size Maximum workspace size in bytes
     * @return Unique pointer to built engine or nullptr on failure
     */
    static std::unique_ptr<nvinfer1::ICudaEngine> buildEngineFromOnnx(
        const std::string& onnx_path,
        const std::string& precision = "FP16",
        int batch_size = 1,
        size_t workspace_size = 1073741824);

    /**
     * @brief Save engine to file
     * @param engine TensorRT engine to save
     * @param engine_path Output file path
     * @return True on success, false on failure
     */
    static bool saveEngine(const nvinfer1::ICudaEngine& engine, const std::string& engine_path);

    /**
     * @brief Load engine from file
     * @param engine_path Path to engine file
     * @param runtime TensorRT runtime instance
     * @return Unique pointer to loaded engine or nullptr on failure
     */
    static std::unique_ptr<nvinfer1::ICudaEngine> loadEngine(
        const std::string& engine_path, nvinfer1::IRuntime& runtime);

    /**
     * @brief Print detailed engine information
     * @param engine TensorRT engine to analyze
     */
    static void printEngineInfo(const nvinfer1::ICudaEngine& engine);
};

/**
 * @brief Utility functions for CUDA and TensorRT operations
 */
namespace utils {
    /**
     * @brief Get size in bytes for a given data type
     */
    size_t getDataTypeSize(nvinfer1::DataType data_type);

    /**
     * @brief Calculate total tensor size in bytes
     */
    size_t calculateTensorSize(const nvinfer1::Dims& dims, nvinfer1::DataType data_type);

    /**
     * @brief Print CUDA device information
     */
    void printCudaDeviceInfo();

    /**
     * @brief Check if CUDA device supports required compute capability
     */
    bool checkCudaCapability(int major_required, int minor_required);

    /**
     * @brief File utility functions
     */
    bool fileExists(const std::string& path);
    std::vector<char> readBinaryFile(const std::string& path);
    bool writeBinaryFile(const std::string& path, const std::vector<char>& data);

    /**
     * @brief Data type conversion utilities
     */
    std::string precisionToString(nvinfer1::DataType data_type);
    nvinfer1::DataType stringToPrecision(const std::string& precision);
}

/**
 * @brief Performance monitoring utilities
 */
class PerformanceMonitor {
public:
    PerformanceMonitor();

    void startTiming(const std::string& operation);
    void endTiming(const std::string& operation);
    void recordInferenceTime(double time_ms);
    void recordMemoryUsage(size_t memory_bytes);

    double getCurrentFPS() const;
    double getAverageInferenceTime() const;
    double getMaxInferenceTime() const;
    double getMinInferenceTime() const;
    size_t getCurrentMemoryUsage() const;

    void printStatistics() const;
    void reset();

private:
    struct TimingData {
        std::chrono::high_resolution_clock::time_point start_time;
        double total_time = 0.0;
        int count = 0;
    };

    std::map<std::string, TimingData> timing_data_;
    std::vector<double> inference_times_;
    size_t current_memory_usage_;
    std::chrono::high_resolution_clock::time_point session_start_;
    int frame_count_;
    mutable std::mutex stats_mutex_;
};

} // namespace lane_detection

#endif // LANE_DETECTION_TENSOR_UTILS_HPP