#include "tensor_utils.hpp"
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <cstring>

namespace lane_detection {

// TensorRT Logger Implementation
void TensorRTLogger::log(Severity severity, const char* msg) noexcept {
    const char* severityStr = getSeverityString(severity);
    if (severity <= Severity::kWARNING) {
        std::cout << "[TensorRT " << severityStr << "] " << msg << std::endl;
    }
}

const char* TensorRTLogger::getSeverityString(Severity severity) {
    switch (severity) {
        case Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
        case Severity::kERROR: return "ERROR";
        case Severity::kWARNING: return "WARNING";
        case Severity::kINFO: return "INFO";
        case Severity::kVERBOSE: return "VERBOSE";
        default: return "UNKNOWN";
    }
}

// Global logger instance
static TensorRTLogger g_logger;

// TensorBuffer Implementation
TensorBuffer::TensorBuffer(size_t size, nvinfer1::DataType data_type)
    : device_ptr_(nullptr), host_ptr_(nullptr), size_(size), data_type_(data_type) {
    if (!allocateMemory()) {
        throw std::runtime_error("Failed to allocate tensor buffer memory");
    }
}

TensorBuffer::~TensorBuffer() {
    deallocateMemory();
}

TensorBuffer::TensorBuffer(TensorBuffer&& other) noexcept
    : device_ptr_(other.device_ptr_), host_ptr_(other.host_ptr_),
      size_(other.size_), data_type_(other.data_type_) {
    other.device_ptr_ = nullptr;
    other.host_ptr_ = nullptr;
    other.size_ = 0;
}

TensorBuffer& TensorBuffer::operator=(TensorBuffer&& other) noexcept {
    if (this != &other) {
        deallocateMemory();
        device_ptr_ = other.device_ptr_;
        host_ptr_ = other.host_ptr_;
        size_ = other.size_;
        data_type_ = other.data_type_;
        
        other.device_ptr_ = nullptr;
        other.host_ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

bool TensorBuffer::allocateMemory() {
    // Allocate pinned host memory
    if (cudaMallocHost(&host_ptr_, size_) != cudaSuccess) {
        std::cerr << "Failed to allocate host memory: " << size_ << " bytes" << std::endl;
        return false;
    }
    
    // Allocate device memory
    if (cudaMalloc(&device_ptr_, size_) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory: " << size_ << " bytes" << std::endl;
        cudaFreeHost(host_ptr_);
        host_ptr_ = nullptr;
        return false;
    }
    
    return true;
}

void TensorBuffer::deallocateMemory() {
    if (host_ptr_) {
        cudaFreeHost(host_ptr_);
        host_ptr_ = nullptr;
    }
    if (device_ptr_) {
        cudaFree(device_ptr_);
        device_ptr_ = nullptr;
    }
}

bool TensorBuffer::hostToDevice(cudaStream_t stream) {
    if (!host_ptr_ || !device_ptr_) return false;
    
    cudaError_t error = cudaMemcpyAsync(device_ptr_, host_ptr_, size_, 
                                       cudaMemcpyHostToDevice, stream);
    return error == cudaSuccess;
}

bool TensorBuffer::deviceToHost(cudaStream_t stream) {
    if (!host_ptr_ || !device_ptr_) return false;
    
    cudaError_t error = cudaMemcpyAsync(host_ptr_, device_ptr_, size_, 
                                       cudaMemcpyDeviceToHost, stream);
    return error == cudaSuccess;
}

bool TensorBuffer::copyFromHost(const void* src, size_t size, cudaStream_t stream) {
    if (!host_ptr_ || size > size_) return false;
    
    std::memcpy(host_ptr_, src, size);
    return hostToDevice(stream);
}

bool TensorBuffer::copyToHost(void* dst, size_t size, cudaStream_t stream) {
    if (!host_ptr_ || size > size_) return false;
    
    if (!deviceToHost(stream)) return false;
    
    if (stream != 0) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }
    
    std::memcpy(dst, host_ptr_, size);
    return true;
}

bool TensorBuffer::zeroInitialize(cudaStream_t stream) {
    if (!device_ptr_) return false;
    
    cudaError_t error = cudaMemsetAsync(device_ptr_, 0, size_, stream);
    return error == cudaSuccess;
}

// EngineUtils Implementation
TensorRTLogger EngineUtils::logger_;

std::unique_ptr<nvinfer1::ICudaEngine> EngineUtils::buildEngineFromOnnx(
    const std::string& onnx_path, const std::string& precision, 
    int batch_size, size_t workspace_size) {
    
    if (!utils::fileExists(onnx_path)) {
        std::cerr << "ONNX file not found: " << onnx_path << std::endl;
        return nullptr;
    }
    
    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger_));
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder" << std::endl;
        return nullptr;
    }
    
    // Create network
    const auto explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));
    if (!network) {
        std::cerr << "Failed to create network" << std::endl;
        return nullptr;
    }
    
    // Create ONNX parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_));
    if (!parser) {
        std::cerr << "Failed to create ONNX parser" << std::endl;
        return nullptr;
    }
    
    // Parse ONNX model
    if (!parser->parseFromFile(onnx_path.c_str(), 
                              static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file: " << onnx_path << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "Parser error: " << parser->getError(i)->desc() << std::endl;
        }
        return nullptr;
    }
    
    // Create builder config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!config) {
        std::cerr << "Failed to create builder config" << std::endl;
        return nullptr;
    }
    
    // Set workspace size
    config->setMaxWorkspaceSize(workspace_size);
    
    // Set precision
    if (precision == "FP16" && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "Using FP16 precision" << std::endl;
    } else if (precision == "INT8" && builder->platformHasFastInt8()) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        std::cout << "Using INT8 precision" << std::endl;
    } else {
        std::cout << "Using FP32 precision" << std::endl;
    }
    
    // Build engine
    std::cout << "Building TensorRT engine (this may take a while)..." << std::endl;
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config));
    
    if (!engine) {
        std::cerr << "Failed to build engine" << std::endl;
        return nullptr;
    }
    
    std::cout << "Successfully built TensorRT engine" << std::endl;
    printEngineInfo(*engine);
    
    return engine;
}

bool EngineUtils::saveEngine(const nvinfer1::ICudaEngine& engine, 
                           const std::string& engine_path) {
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        engine.serialize());
    if (!serialized) {
        std::cerr << "Failed to serialize engine" << std::endl;
        return false;
    }
    
    std::ofstream file(engine_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << engine_path << std::endl;
        return false;
    }
    
    file.write(static_cast<const char*>(serialized->data()), serialized->size());
    file.close();
    
    std::cout << "Saved engine to: " << engine_path << std::endl;
    return true;
}

std::unique_ptr<nvinfer1::ICudaEngine> EngineUtils::loadEngine(
    const std::string& engine_path, nvinfer1::IRuntime& runtime) {
    
    if (!utils::fileExists(engine_path)) {
        std::cerr << "Engine file not found: " << engine_path << std::endl;
        return nullptr;
    }
    
    std::vector<char> engineData = utils::readBinaryFile(engine_path);
    if (engineData.empty()) {
        std::cerr << "Failed to read engine file" << std::endl;
        return nullptr;
    }
    
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime.deserializeCudaEngine(engineData.data(), engineData.size()));
    
    if (!engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return nullptr;
    }
    
    std::cout << "Loaded engine from: " << engine_path << std::endl;
    printEngineInfo(*engine);
    
    return engine;
}

void EngineUtils::printEngineInfo(const nvinfer1::ICudaEngine& engine) {
    std::cout << "=== Engine Information ===" << std::endl;
    std::cout << "Number of bindings: " << engine.getNbBindings() << std::endl;
    
    for (int i = 0; i < engine.getNbBindings(); ++i) {
        const char* name = engine.getBindingName(i);
        auto dims = engine.getBindingDimensions(i);
        auto dtype = engine.getBindingDataType(i);
        bool isInput = engine.bindingIsInput(i);
        
        std::cout << "Binding " << i << ": " << name 
                  << (isInput ? " [INPUT]" : " [OUTPUT]") << std::endl;
        std::cout << "  Shape: ";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << "x";
        }
        std::cout << std::endl;
        std::cout << "  DataType: " << utils::precisionToString(dtype) << std::endl;
        std::cout << "  Size: " << utils::calculateTensorSize(dims, dtype) << " bytes" << std::endl;
    }
    std::cout << "=========================" << std::endl;
}

// Utility functions
namespace utils {
    size_t getDataTypeSize(nvinfer1::DataType data_type) {
        switch (data_type) {
            case nvinfer1::DataType::kFLOAT: return sizeof(float);
            case nvinfer1::DataType::kHALF: return sizeof(uint16_t);
            case nvinfer1::DataType::kINT8: return sizeof(int8_t);
            case nvinfer1::DataType::kINT32: return sizeof(int32_t);
            case nvinfer1::DataType::kBOOL: return sizeof(bool);
            default: return 0;
        }
    }
    
    size_t calculateTensorSize(const nvinfer1::Dims& dims, nvinfer1::DataType data_type) {
        size_t size = getDataTypeSize(data_type);
        for (int i = 0; i < dims.nbDims; ++i) {
            size *= dims.d[i];
        }
        return size;
    }
    
    void printCudaDeviceInfo() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        
        std::cout << "=== CUDA Device Information ===" << std::endl;
        std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
        
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            
            std::cout << "Device " << i << ": " << prop.name << std::endl;
            std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "  Total Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
            std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        }
        std::cout << "===============================" << std::endl;
    }
    
    bool checkCudaCapability(int major_required, int minor_required) {
        int device;
        cudaGetDevice(&device);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        return (prop.major > major_required) || 
               (prop.major == major_required && prop.minor >= minor_required);
    }
    
    bool fileExists(const std::string& path) {
        std::ifstream file(path);
        return file.good();
    }
    
    std::vector<char> readBinaryFile(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) return {};
        
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            return {};
        }
        
        return buffer;
    }
    
    bool writeBinaryFile(const std::string& path, const std::vector<char>& data) {
        std::ofstream file(path, std::ios::binary);
        if (!file) return false;
        
        file.write(data.data(), data.size());
        return file.good();
    }
    
    std::string precisionToString(nvinfer1::DataType data_type) {
        switch (data_type) {
            case nvinfer1::DataType::kFLOAT: return "FP32";
            case nvinfer1::DataType::kHALF: return "FP16";
            case nvinfer1::DataType::kINT8: return "INT8";
            case nvinfer1::DataType::kINT32: return "INT32";
            case nvinfer1::DataType::kBOOL: return "BOOL";
            default: return "UNKNOWN";
        }
    }
    
    nvinfer1::DataType stringToPrecision(const std::string& precision) {
        if (precision == "FP32") return nvinfer1::DataType::kFLOAT;
        if (precision == "FP16") return nvinfer1::DataType::kHALF;
        if (precision == "INT8") return nvinfer1::DataType::kINT8;
        if (precision == "INT32") return nvinfer1::DataType::kINT32;
        if (precision == "BOOL") return nvinfer1::DataType::kBOOL;
        return nvinfer1::DataType::kFLOAT; // Default
    }
}

} // namespace lane_detection