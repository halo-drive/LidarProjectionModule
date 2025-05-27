#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <curand_kernel.h>
#include <math.h>

namespace lidar_processing {
namespace cuda {

// CUDA-compatible point structure
struct CudaPoint {
    float x, y, z;
    float intensity;
    uint16_t ring;
    float range;
    uint8_t sensor_id;
};

struct PlaneCoefficients {
    float a, b, c, d;
};

// CUDA kernel for calculating point ranges and rings
__global__ void calculatePointMetrics(CudaPoint* points, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;
    
    CudaPoint& point = points[idx];
    
    // Calculate range
    point.range = sqrtf(point.x * point.x + point.y * point.y + point.z * point.z);
    
    // Calculate vertical angle for ring estimation
    float vertical_angle = atan2f(point.z, sqrtf(point.x * point.x + point.y * point.y));
    float vertical_angle_deg = vertical_angle * 180.0f / M_PI;
    
    // VLP-16 vertical angles
    const float vlp16_angles[16] = {
        -15.0f, -13.0f, -11.0f, -9.0f, -7.0f, -5.0f, -3.0f, -1.0f,
        1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f
    };
    
    // Find closest ring
    int closest_ring = 0;
    float min_diff = fabsf(vertical_angle_deg - vlp16_angles[0]);
    
    for (int i = 1; i < 16; ++i) {
        float diff = fabsf(vertical_angle_deg - vlp16_angles[i]);
        if (diff < min_diff) {
            min_diff = diff;
            closest_ring = i;
        }
    }
    
    point.ring = static_cast<uint16_t>(closest_ring);
}

// CUDA kernel for range filtering
__global__ void rangeFilter(const CudaPoint* input, CudaPoint* output, bool* valid_mask,
                           int num_points, float min_range, float max_range, 
                           float min_height, float max_height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;
    
    const CudaPoint& point = input[idx];
    
    // Check validity
    bool is_valid = isfinite(point.x) && isfinite(point.y) && isfinite(point.z) &&
                   point.range >= min_range && point.range <= max_range &&
                   point.z >= min_height && point.z <= max_height;
    
    valid_mask[idx] = is_valid;
    
    if (is_valid) {
        output[idx] = point;
    }
}

// CUDA kernel for RANSAC plane fitting
__global__ void ransacPlaneKernel(const CudaPoint* points, int num_points,
                                 PlaneCoefficients* best_plane, float* best_score,
                                 int max_iterations, float distance_threshold,
                                 curandState* rand_states) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    __shared__ PlaneCoefficients shared_plane;
    __shared__ float shared_score;
    
    if (threadIdx.x == 0) {
        shared_score = 0.0f;
    }
    __syncthreads();
    
    float local_best_score = 0.0f;
    PlaneCoefficients local_best_plane = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Each thread performs multiple RANSAC iterations
    for (int iter = tid; iter < max_iterations; iter += stride) {
        curandState local_state = rand_states[tid];
        
        // Randomly select 3 points
        int idx1 = curand(&local_state) % num_points;
        int idx2 = curand(&local_state) % num_points;
        int idx3 = curand(&local_state) % num_points;
        
        // Ensure different points
        while (idx2 == idx1) idx2 = curand(&local_state) % num_points;
        while (idx3 == idx1 || idx3 == idx2) idx3 = curand(&local_state) % num_points;
        
        const CudaPoint& p1 = points[idx1];
        const CudaPoint& p2 = points[idx2];
        const CudaPoint& p3 = points[idx3];
        
        // Calculate plane coefficients using cross product
        float v1x = p2.x - p1.x, v1y = p2.y - p1.y, v1z = p2.z - p1.z;
        float v2x = p3.x - p1.x, v2y = p3.y - p1.y, v2z = p3.z - p1.z;
        
        // Normal vector (cross product)
        float nx = v1y * v2z - v1z * v2y;
        float ny = v1z * v2x - v1x * v2z;
        float nz = v1x * v2y - v1y * v2x;
        
        // Normalize
        float norm = sqrtf(nx * nx + ny * ny + nz * nz);
        if (norm < 1e-6f) continue;
        
        nx /= norm;
        ny /= norm;
        nz /= norm;
        
        // Calculate d coefficient
        float d = -(nx * p1.x + ny * p1.y + nz * p1.z);
        
        PlaneCoefficients plane = {nx, ny, nz, d};
        
        // Count inliers
        int inlier_count = 0;
        for (int i = 0; i < num_points; ++i) {
            const CudaPoint& pt = points[i];
            float distance = fabsf(plane.a * pt.x + plane.b * pt.y + plane.c * pt.z + plane.d);
            
            if (distance < distance_threshold) {
                inlier_count++;
            }
        }
        
        float score = static_cast<float>(inlier_count) / num_points;
        
        if (score > local_best_score) {
            local_best_score = score;
            local_best_plane = plane;
        }
        
        rand_states[tid] = local_state;
    }
    
    // Reduce to find best plane across all threads
    __syncthreads();
    
    if (local_best_score > shared_score) {
        atomicExch(&shared_score, local_best_score);
        shared_plane = local_best_plane;
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0 && shared_score > *best_score) {
        atomicExch(best_score, shared_score);
        *best_plane = shared_plane;
    }
}

// CUDA kernel for ground point classification
__global__ void classifyGroundPoints(const CudaPoint* points, bool* ground_mask, 
                                   float* confidence_scores, int num_points,
                                   PlaneCoefficients plane, float distance_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;
    
    const CudaPoint& point = points[idx];
    
    // Calculate distance to plane
    float distance = fabsf(plane.a * point.x + plane.b * point.y + 
                          plane.c * point.z + plane.d);
    
    // Ground classification based on distance
    bool is_ground = distance < distance_threshold;
    ground_mask[idx] = is_ground;
    
    // Calculate confidence score
    float confidence = expf(-distance / distance_threshold);
    
    // Additional confidence factors
    // Height constraint
    if (point.z < -2.5f || point.z > 0.5f) {
        confidence *= 0.5f;
    }
    
    // Normal alignment (plane normal should point roughly upward)
    float normal_z_component = fabsf(plane.c);
    confidence *= normal_z_component;
    
    confidence_scores[idx] = fminf(1.0f, fmaxf(0.0f, confidence));
}

// CUDA kernel for ring-based ground extraction
__global__ void ringBasedGroundExtraction(const CudaPoint* points, bool* ground_mask,
                                        int num_points, float height_threshold,
                                        int start_ring, int end_ring) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;
    
    const CudaPoint& point = points[idx];
    
    // Check if point is in valid ring range
    if (point.ring < start_ring || point.ring > end_ring) {
        ground_mask[idx] = false;
        return;
    }
    
    // Simple ring-based classification
    // Points with similar height in the same ring are likely ground
    bool is_ground = false;
    
    // Check neighboring points in the same ring
    for (int i = 0; i < num_points; ++i) {
        if (i == idx) continue;
        
        const CudaPoint& neighbor = points[i];
        
        // Same ring and close in distance
        if (neighbor.ring == point.ring) {
            float dist = sqrtf((point.x - neighbor.x) * (point.x - neighbor.x) +
                              (point.y - neighbor.y) * (point.y - neighbor.y));
            
            if (dist < 2.0f) { // Within 2 meters
                float height_diff = fabsf(point.z - neighbor.z);
                if (height_diff < height_threshold) {
                    is_ground = true;
                    break;
                }
            }
        }
    }
    
    ground_mask[idx] = is_ground;
}

// CUDA kernel for morphological filtering
__global__ void morphologicalFilter(const CudaPoint* points, const bool* input_mask,
                                  bool* output_mask, int num_points,
                                  float connectivity_radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;
    
    if (!input_mask[idx]) {
        output_mask[idx] = false;
        return;
    }
    
    const CudaPoint& point = points[idx];
    int neighbor_count = 0;
    
    // Count neighboring ground points
    for (int i = 0; i < num_points; ++i) {
        if (i == idx || !input_mask[i]) continue;
        
        const CudaPoint& neighbor = points[i];
        float dist = sqrtf((point.x - neighbor.x) * (point.x - neighbor.x) +
                          (point.y - neighbor.y) * (point.y - neighbor.y) +
                          (point.z - neighbor.z) * (point.z - neighbor.z));
        
        if (dist < connectivity_radius) {
            neighbor_count++;
        }
    }
    
    // Keep point if it has enough neighbors
    output_mask[idx] = (neighbor_count >= 2);
}

// CUDA kernel for voxel grid downsampling
__global__ void voxelGridDownsample(const CudaPoint* input, CudaPoint* output,
                                   int* output_indices, int num_points,
                                   float voxel_size, int* voxel_map,
                                   int grid_size_x, int grid_size_y, int grid_size_z,
                                   float min_x, float min_y, float min_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;
    
    const CudaPoint& point = input[idx];
    
    // Calculate voxel coordinates
    int vx = static_cast<int>((point.x - min_x) / voxel_size);
    int vy = static_cast<int>((point.y - min_y) / voxel_size);
    int vz = static_cast<int>((point.z - min_z) / voxel_size);
    
    // Bounds check
    if (vx < 0 || vx >= grid_size_x || vy < 0 || vy >= grid_size_y || 
        vz < 0 || vz >= grid_size_z) {
        return;
    }
    
    // Calculate voxel index
    int voxel_idx = vz * grid_size_x * grid_size_y + vy * grid_size_x + vx;
    
    // Atomic operation to claim this voxel
    int old_val = atomicCAS(&voxel_map[voxel_idx], -1, idx);
    
    if (old_val == -1) {
        // This thread claimed the voxel
        int output_idx = atomicAdd(&output_indices[0], 1);
        output[output_idx] = point;
    }
}

// CUDA kernel initialization for random states
__global__ void initRandomStates(curandState* states, unsigned long seed, int num_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_states) return;
    
    curand_init(seed, idx, 0, &states[idx]);
}

// Host function to perform CUDA-accelerated ground extraction
extern "C" {

bool cudaGroundExtraction(const CudaPoint* h_points, int num_points,
                         bool* h_ground_mask, float* h_confidence_scores,
                         PlaneCoefficients* h_plane_coeffs,
                         float distance_threshold, int max_iterations) {
    
    // Device memory allocation
    CudaPoint* d_points;
    bool* d_ground_mask;
    float* d_confidence_scores;
    PlaneCoefficients* d_best_plane;
    float* d_best_score;
    curandState* d_rand_states;
    
    cudaMalloc(&d_points, num_points * sizeof(CudaPoint));
    cudaMalloc(&d_ground_mask, num_points * sizeof(bool));
    cudaMalloc(&d_confidence_scores, num_points * sizeof(float));
    cudaMalloc(&d_best_plane, sizeof(PlaneCoefficients));
    cudaMalloc(&d_best_score, sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_points, h_points, num_points * sizeof(CudaPoint), cudaMemcpyHostToDevice);
    
    // Initialize random states
    int num_threads = 256;
    int num_blocks = (num_points + num_threads - 1) / num_threads;
    
    cudaMalloc(&d_rand_states, num_threads * num_blocks * sizeof(curandState));
    initRandomStates<<<num_blocks, num_threads>>>(d_rand_states, time(NULL), num_threads * num_blocks);
    
    // Initialize best score
    float initial_score = 0.0f;
    cudaMemcpy(d_best_score, &initial_score, sizeof(float), cudaMemcpyHostToDevice);
    
    // Perform RANSAC plane fitting
    ransacPlaneKernel<<<num_blocks, num_threads>>>(
        d_points, num_points, d_best_plane, d_best_score,
        max_iterations, distance_threshold, d_rand_states);
    
    // Copy best plane back to host
    cudaMemcpy(h_plane_coeffs, d_best_plane, sizeof(PlaneCoefficients), cudaMemcpyDeviceToHost);
    
    // Classify ground points
    classifyGroundPoints<<<num_blocks, num_threads>>>(
        d_points, d_ground_mask, d_confidence_scores, num_points,
        *h_plane_coeffs, distance_threshold);
    
    // Copy results back to host
    cudaMemcpy(h_ground_mask, d_ground_mask, num_points * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_confidence_scores, d_confidence_scores, num_points * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Clean up device memory
    cudaFree(d_points);
    cudaFree(d_ground_mask);
    cudaFree(d_confidence_scores);
    cudaFree(d_best_plane);
    cudaFree(d_best_score);
    cudaFree(d_rand_states);
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return false;
    }
    
    return true;
}

bool cudaVoxelGridFilter(const CudaPoint* h_input, int num_input,
                        CudaPoint* h_output, int* h_num_output,
                        float voxel_size) {
    
    // Find bounding box
    float min_x = h_input[0].x, max_x = h_input[0].x;
    float min_y = h_input[0].y, max_y = h_input[0].y;
    float min_z = h_input[0].z, max_z = h_input[0].z;
    
    for (int i = 1; i < num_input; ++i) {
        min_x = fminf(min_x, h_input[i].x);
        max_x = fmaxf(max_x, h_input[i].x);
        min_y = fminf(min_y, h_input[i].y);
        max_y = fmaxf(max_y, h_input[i].y);
        min_z = fminf(min_z, h_input[i].z);
        max_z = fmaxf(max_z, h_input[i].z);
    }
    
    // Calculate grid dimensions
    int grid_size_x = static_cast<int>((max_x - min_x) / voxel_size) + 1;
    int grid_size_y = static_cast<int>((max_y - min_y) / voxel_size) + 1;
    int grid_size_z = static_cast<int>((max_z - min_z) / voxel_size) + 1;
    
    // Device memory allocation
    CudaPoint* d_input;
    CudaPoint* d_output;
    int* d_output_indices;
    int* d_voxel_map;
    
    cudaMalloc(&d_input, num_input * sizeof(CudaPoint));
    cudaMalloc(&d_output, num_input * sizeof(CudaPoint)); // Max possible output
    cudaMalloc(&d_output_indices, sizeof(int));
    cudaMalloc(&d_voxel_map, grid_size_x * grid_size_y * grid_size_z * sizeof(int));
    
    // Initialize voxel map
    cudaMemset(d_voxel_map, -1, grid_size_x * grid_size_y * grid_size_z * sizeof(int));
    cudaMemset(d_output_indices, 0, sizeof(int));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, num_input * sizeof(CudaPoint), cudaMemcpyHostToDevice);
    
    // Perform voxel grid filtering
    int num_threads = 256;
    int num_blocks = (num_input + num_threads - 1) / num_threads;
    
    voxelGridDownsample<<<num_blocks, num_threads>>>(
        d_input, d_output, d_output_indices, num_input, voxel_size,
        d_voxel_map, grid_size_x, grid_size_y, grid_size_z,
        min_x, min_y, min_z);
    
    // Copy results back
    cudaMemcpy(h_num_output, d_output_indices, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output, d_output, *h_num_output * sizeof(CudaPoint), cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_indices);
    cudaFree(d_voxel_map);
    
    return true;
}

} // extern "C"

} // namespace cuda
} // namespace lidar_processing