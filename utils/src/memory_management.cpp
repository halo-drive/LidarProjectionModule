/**
 * @file memory_management.cpp
 * @brief Implementation of memory management system
 */

#include "utils/include/memory_management.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <cstring>

namespace lane_fusion {
namespace utils {

// Helper function for CUDA error checking
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
        abort(); \
    } \
} while(0)

// MemoryPool implementation

MemoryPool::MemoryPool(MemoryType type, const MemoryPoolConfig& config)
    : type_(type), config_(config), stats_() {
}

MemoryPool::~MemoryPool() {
    // Free all allocated blocks
    for (auto& block : blocks_) {
        if (block.ptr) {
            deallocateSystem(block.ptr);
        }
    }
}

void* MemoryPool::allocate(size_t size, size_t alignment) {
    if (size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Find suitable existing block
    size_t blockIndex = findSuitableBlock(size, alignment);

    // If no suitable block, allocate new one
    if (blockIndex == (size_t)-1) {
        blockIndex = allocateNewBlock(size);
    }

    // Mark block as in use
    MemoryBlock& block = blocks_[blockIndex];
    block.inUse = true;

    // Update stats
    if (config_.trackUsage) {
        stats_.currentUsage += block.size;
        stats_.peakUsage = std::max(stats_.peakUsage, stats_.currentUsage);
        stats_.allocationCount++;
    }

    return block.ptr;
}

void MemoryPool::deallocate(void* ptr) {
    if (!ptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Find the block
    auto it = std::find_if(blocks_.begin(), blocks_.end(),
        [ptr](const MemoryBlock& block) { return block.ptr == ptr; });

    if (it != blocks_.end()) {
        // Update stats
        if (config_.trackUsage) {
            stats_.currentUsage -= it->size;
            stats_.releaseCount++;
        }

        // Mark as available
        it->inUse = false;
    } else {
        std::cerr << "Warning: Attempt to deallocate memory not owned by this pool" << std::endl;
    }
}

MemoryType MemoryPool::getType() const {
    return type_;
}

MemoryStats MemoryPool::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void MemoryPool::trim(size_t threshold) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Free unused blocks larger than threshold
    for (auto it = blocks_.begin(); it != blocks_.end();) {
        if (!it->inUse && it->size >= threshold) {
            deallocateSystem(it->ptr);
            stats_.totalPoolSize -= it->size;
            it = blocks_.erase(it);
        } else {
            ++it;
        }
    }
}

size_t MemoryPool::allocateNewBlock(size_t size) {
    // Round size up to nearest multiple of initial block size
    size_t blockSize = std::max(size, config_.initialBlockSize);

    // Check if we would exceed max pool size
    if (stats_.totalPoolSize + blockSize > config_.maxPoolSize) {
        // Try to trim first
        trim(0);

        // If still exceeding, that's an error
        if (stats_.totalPoolSize + blockSize > config_.maxPoolSize) {
            throw std::bad_alloc();
        }
    }

    // Allocate new memory
    void* ptr = allocateSystem(blockSize);
    if (!ptr) {
        throw std::bad_alloc();
    }

    // Create new block
    MemoryBlock block;
    block.ptr = ptr;
    block.size = blockSize;
    block.inUse = false;
    block.type = type_;

    // Update stats
    stats_.totalPoolSize += blockSize;
    stats_.totalAllocated += blockSize;

    // Add to blocks
    blocks_.push_back(block);
    return blocks_.size() - 1;
}

size_t MemoryPool::findSuitableBlock(size_t size, size_t alignment) {
    // First-fit strategy for simplicity
    for (size_t i = 0; i < blocks_.size(); i++) {
        if (!blocks_[i].inUse && blocks_[i].size >= size) {
            // Check alignment
            uintptr_t addr = reinterpret_cast<uintptr_t>(blocks_[i].ptr);
            if (addr % alignment == 0) {
                return i;
            }
        }
    }

    return (size_t)-1; // Not found
}

void* MemoryPool::allocateSystem(size_t size) {
    void* ptr = nullptr;

    switch (type_) {
    case MemoryType::HOST:
        // Standard aligned allocation
        #ifdef _WIN32
        ptr = _aligned_malloc(size, 16);
        #else
        if (posix_memalign(&ptr, 16, size) != 0) {
            ptr = nullptr;
        }
        #endif
        break;

    case MemoryType::DEVICE:
        // CUDA device memory
        CUDA_CHECK(cudaMalloc(&ptr, size));
        break;

    case MemoryType::UNIFIED:
        // CUDA managed/unified memory
        CUDA_CHECK(cudaMallocManaged(&ptr, size));
        break;

    case MemoryType::PINNED:
        // CUDA pinned memory for faster transfers
        CUDA_CHECK(cudaMallocHost(&ptr, size));
        break;
    }

    return ptr;
}

void MemoryPool::deallocateSystem(void* ptr) {
    if (!ptr) return;

    switch (type_) {
    case MemoryType::HOST:
        #ifdef _WIN32
        _aligned_free(ptr);
        #else
        free(ptr);
        #endif
        break;

    case MemoryType::DEVICE:
        CUDA_CHECK(cudaFree(ptr));
        break;

    case MemoryType::UNIFIED:
        CUDA_CHECK(cudaFree(ptr));
        break;

    case MemoryType::PINNED:
        CUDA_CHECK(cudaFreeHost(ptr));
        break;
    }
}

// MemoryManager implementation

MemoryManager& MemoryManager::getInstance() {
    static MemoryManager instance;
    return instance;
}

MemoryManager::MemoryManager() {
    // Initialize default pools
    pools_[MemoryType::HOST] = std::make_unique<MemoryPool>(MemoryType::HOST);

    // Only initialize CUDA pools if CUDA is available
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err == cudaSuccess && deviceCount > 0) {
        pools_[MemoryType::DEVICE] = std::make_unique<MemoryPool>(MemoryType::DEVICE);
        pools_[MemoryType::UNIFIED] = std::make_unique<MemoryPool>(MemoryType::UNIFIED);
        pools_[MemoryType::PINNED] = std::make_unique<MemoryPool>(MemoryType::PINNED);
    } else {
        std::cerr << "Warning: No CUDA devices available, GPU memory pools disabled" << std::endl;
    }
}

MemoryManager::~MemoryManager() {
    // pools_ will automatically clean up via unique_ptr
}

void* MemoryManager::allocate(MemoryType type, size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = pools_.find(type);
    if (it == pools_.end()) {
        throw std::runtime_error("Memory pool for requested type does not exist");
    }

    void* ptr = it->second->allocate(size, alignment);
    if (ptr) {
        allocations_[ptr] = type;
    }

    return ptr;
}

void MemoryManager::deallocate(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        MemoryType type = it->second;
        pools_[type]->deallocate(ptr);
        allocations_.erase(it);
    } else {
        std::cerr << "Warning: Attempt to deallocate unknown memory" << std::endl;
    }
}

MemoryStats MemoryManager::getStats(MemoryType type) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = pools_.find(type);
    if (it != pools_.end()) {
        return it->second->getStats();
    } else {
        return MemoryStats{};
    }
}

void MemoryManager::trimAll(size_t threshold) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& pair : pools_) {
        pair.second->trim(threshold);
    }
}

void MemoryManager::configurePool(MemoryType type, const MemoryPoolConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = pools_.find(type);
    if (it == pools_.end()) {
        // Create new pool with config
        pools_[type] = std::make_unique<MemoryPool>(type, config);
    } else {
        // For existing pools, we'd need to implement a reconfigure method
        // or recreate the pool, which would lose existing allocations
        std::cerr << "Warning: Reconfiguring existing pools is not supported" << std::endl;
    }
}

} // namespace utils
} // namespace lane_fusion