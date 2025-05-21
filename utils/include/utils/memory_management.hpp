/**
 * @file memory_management.hpp
 * @brief Memory management utilities optimized for Jetson AGX Orin
 *
 * This file defines a memory management system designed for efficient
 * operation on the Jetson AGX Orin platform. It provides:
 * - CUDA-aware memory pooling
 * - Zero-copy memory transfers where possible
 * - Aligned memory allocation for optimized access
 * - Tracking and profiling of memory usage
 */

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstddef>
#include <cuda_runtime.h>

namespace lane_fusion {
namespace utils {

/**
 * @brief Memory location type
 */
enum class MemoryType {
    HOST,           // CPU memory
    DEVICE,         // GPU memory
    UNIFIED,        // Unified memory (accessible from both CPU and GPU)
    PINNED          // Pinned/page-locked memory
};

/**
 * @brief Memory pool settings
 */
struct MemoryPoolConfig {
    size_t initialBlockSize = 1024 * 1024;  // 1MB initial block
    size_t growthFactor = 2;                // Double size when growing
    size_t maxPoolSize = 1024 * 1024 * 512; // 512MB max size
    bool trackUsage = true;                 // Enable usage tracking
};

/**
 * @brief Memory tracking statistics
 */
struct MemoryStats {
    size_t totalAllocated = 0;
    size_t totalPoolSize = 0;
    size_t peakUsage = 0;
    size_t currentUsage = 0;
    size_t allocationCount = 0;
    size_t releaseCount = 0;
};

/**
 * @brief Memory block managed by the pool
 */
struct MemoryBlock {
    void* ptr = nullptr;
    size_t size = 0;
    bool inUse = false;
    MemoryType type = MemoryType::HOST;
};

/**
 * @class MemoryPool
 * @brief Pool-based memory manager for efficient memory reuse
 *
 * This class manages memory allocation/deallocation with pooling to
 * reduce fragmentation and system calls. It's optimized for the
 * heterogeneous memory architecture of Jetson AGX Orin.
 */
class MemoryPool {
public:
    /**
     * @brief Construct a memory pool
     * @param type Type of memory to manage
     * @param config Pool configuration parameters
     */
    explicit MemoryPool(MemoryType type, const MemoryPoolConfig& config = MemoryPoolConfig());

    /**
     * @brief Destructor - frees all allocated memory
     */
    ~MemoryPool();

    /**
     * @brief Allocate memory from the pool
     * @param size Size in bytes to allocate
     * @param alignment Memory alignment (default: 16 bytes)
     * @return Pointer to allocated memory
     */
    void* allocate(size_t size, size_t alignment = 16);

    /**
     * @brief Return memory to the pool
     * @param ptr Pointer to memory previously allocated from this pool
     */
    void deallocate(void* ptr);

    /**
     * @brief Get memory type of this pool
     * @return Memory type
     */
    MemoryType getType() const;

    /**
     * @brief Get memory usage statistics
     * @return Current memory statistics
     */
    MemoryStats getStats() const;

    /**
     * @brief Release unused memory back to the system
     * @param threshold Release blocks with size >= threshold
     */
    void trim(size_t threshold = 0);

private:
    MemoryType type_;
    MemoryPoolConfig config_;
    std::vector<MemoryBlock> blocks_;
    MemoryStats stats_;
    mutable std::mutex mutex_;

    /**
     * @brief Allocate a new memory block from the system
     * @param size Minimum size needed
     * @return Index of the new block in blocks_ vector
     */
    size_t allocateNewBlock(size_t size);

    /**
     * @brief Find a suitable existing block
     * @param size Size needed
     * @param alignment Required alignment
     * @return Index of suitable block or -1 if none found
     */
    size_t findSuitableBlock(size_t size, size_t alignment);

    /**
     * @brief Actually allocate memory from the system based on type
     * @param size Size to allocate
     * @return Raw pointer to allocated memory
     */
    void* allocateSystem(size_t size);

    /**
     * @brief Free system memory based on type
     * @param ptr Pointer to free
     */
    void deallocateSystem(void* ptr);
};

/**
 * @class MemoryManager
 * @brief Singleton manager for all memory pools
 *
 * This class provides global access to memory pools for different
 * memory types, managing their lifecycle and providing a unified
 * interface for memory allocation/deallocation.
 */
class MemoryManager {
public:
    /**
     * @brief Get singleton instance
     * @return Reference to singleton instance
     */
    static MemoryManager& getInstance();

    /**
     * @brief Allocate memory of specified type
     * @param type Memory type
     * @param size Size in bytes
     * @param alignment Memory alignment (default: 16)
     * @return Pointer to allocated memory
     */
    void* allocate(MemoryType type, size_t size, size_t alignment = 16);

    /**
     * @brief Deallocate memory
     * @param ptr Pointer to memory previously allocated by this manager
     */
    void deallocate(void* ptr);

    /**
     * @brief Get memory stats for a specific memory type
     * @param type Memory type
     * @return Memory statistics
     */
    MemoryStats getStats(MemoryType type) const;

    /**
     * @brief Trim all memory pools
     * @param threshold Size threshold for trimming
     */
    void trimAll(size_t threshold = 0);

    /**
     * @brief Configure memory pool
     * @param type Memory type
     * @param config Pool configuration
     */
    void configurePool(MemoryType type, const MemoryPoolConfig& config);

private:
    MemoryManager();
    ~MemoryManager();

    // Disable copy/move
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    MemoryManager(MemoryManager&&) = delete;
    MemoryManager& operator=(MemoryManager&&) = delete;

    std::unordered_map<MemoryType, std::unique_ptr<MemoryPool>> pools_;
    std::unordered_map<void*, MemoryType> allocations_;
    mutable std::mutex mutex_;
};

// Convenience functions for global memory management

/**
 * @brief Allocate host memory
 * @param size Size in bytes
 * @param alignment Memory alignment
 * @return Pointer to allocated memory
 */
inline void* allocateHost(size_t size, size_t alignment = 16) {
    return MemoryManager::getInstance().allocate(MemoryType::HOST, size, alignment);
}

/**
 * @brief Allocate device (GPU) memory
 * @param size Size in bytes
 * @param alignment Memory alignment
 * @return Pointer to allocated memory
 */
inline void* allocateDevice(size_t size, size_t alignment = 16) {
    return MemoryManager::getInstance().allocate(MemoryType::DEVICE, size, alignment);
}

/**
 * @brief Allocate unified memory (accessible from both CPU and GPU)
 * @param size Size in bytes
 * @param alignment Memory alignment
 * @return Pointer to allocated memory
 */
inline void* allocateUnified(size_t size, size_t alignment = 16) {
    return MemoryManager::getInstance().allocate(MemoryType::UNIFIED, size, alignment);
}

/**
 * @brief Allocate pinned memory (page-locked for faster transfers)
 * @param size Size in bytes
 * @param alignment Memory alignment
 * @return Pointer to allocated memory
 */
inline void* allocatePinned(size_t size, size_t alignment = 16) {
    return MemoryManager::getInstance().allocate(MemoryType::PINNED, size, alignment);
}

/**
 * @brief Deallocate memory allocated by one of the allocate functions
 * @param ptr Pointer to memory
 */
inline void deallocate(void* ptr) {
    if (ptr) {
        MemoryManager::getInstance().deallocate(ptr);
    }
}

/**
 * @brief Smart pointer deleter for memory allocated from the pool
 */
struct PoolDeleter {
    void operator()(void* ptr) const {
        deallocate(ptr);
    }
};

/**
 * @brief Create a unique_ptr with pool-allocated memory
 * @tparam T Type of object
 * @tparam Args Constructor argument types
 * @param type Memory type
 * @param args Constructor arguments
 * @return unique_ptr to T with PoolDeleter
 */
template<typename T, typename... Args>
std::unique_ptr<T, PoolDeleter> makeUnique(MemoryType type, Args&&... args) {
    void* memory = MemoryManager::getInstance().allocate(type, sizeof(T), alignof(T));
    T* object = new(memory) T(std::forward<Args>(args)...);
    return std::unique_ptr<T, PoolDeleter>(object);
}

} // namespace utils
} // namespace lane_fusion