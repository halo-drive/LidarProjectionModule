// camera_stitching/cuda/image_kernels.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

#include <cfloat>
#include <cmath>

namespace camera_stitching {

// Constants for optimized memory access
#define BLOCK_SIZE_16 16
#define BLOCK_SIZE_32 32
#define WARP_SIZE 32
#define MAX_FEATURES_PER_BLOCK 256
#define ORB_DESCRIPTOR_SIZE 32

// Device utility functions
__device__ __forceinline__ float bilinearInterpolation(
    const uint8_t* image, float x, float y, int width, int height, int channels, int channel) {
    
    int x0 = __float2int_rd(x);
    int y0 = __float2int_rd(y);
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    
    float dx = x - x0;
    float dy = y - y0;
    
    // Ensure bounds
    x0 = max(0, min(x0, width - 1));
    y0 = max(0, min(y0, height - 1));
    x1 = max(0, min(x1, width - 1));
    y1 = max(0, min(y1, height - 1));
    
    int idx00 = (y0 * width + x0) * channels + channel;
    int idx01 = (y0 * width + x1) * channels + channel;
    int idx10 = (y1 * width + x0) * channels + channel;
    int idx11 = (y1 * width + x1) * channels + channel;
    
    float val00 = image[idx00];
    float val01 = image[idx01];
    float val10 = image[idx10];
    float val11 = image[idx11];
    
    float val0 = val00 * (1.0f - dx) + val01 * dx;
    float val1 = val10 * (1.0f - dx) + val11 * dx;
    
    return val0 * (1.0f - dy) + val1 * dy;
}

__device__ __forceinline__ void applyHomographyTransform(
    float x, float y, const float* homography, float& out_x, float& out_y) {
    
    float w = homography[6] * x + homography[7] * y + homography[8];
    if (fabsf(w) > 1e-7f) {
        out_x = (homography[0] * x + homography[1] * y + homography[2]) / w;
        out_y = (homography[3] * x + homography[4] * y + homography[5]) / w;
    } else {
        out_x = x;
        out_y = y;
    }
}

__device__ __forceinline__ int hammingDistance(const uint8_t* desc1, const uint8_t* desc2, int length) {
    int distance = 0;
    for (int i = 0; i < length; ++i) {
        distance += __popc(desc1[i] ^ desc2[i]);
    }
    return distance;
}

// Image preprocessing kernels
__global__ void undistortImageKernel(
    const uint8_t* d_input, uint8_t* d_output,
    const float* d_camera_matrix, const float* d_dist_coeffs,
    int width, int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Camera matrix elements
    float fx = d_camera_matrix[0];
    float fy = d_camera_matrix[4];
    float cx = d_camera_matrix[2];
    float cy = d_camera_matrix[5];
    
    // Distortion coefficients (assuming 5 coefficients: k1, k2, p1, p2, k3)
    float k1 = d_dist_coeffs[0];
    float k2 = d_dist_coeffs[1];
    float p1 = d_dist_coeffs[2];
    float p2 = d_dist_coeffs[3];
    float k3 = (d_dist_coeffs != nullptr) ? d_dist_coeffs[4] : 0.0f;
    
    // Normalize coordinates
    float x_norm = (x - cx) / fx;
    float y_norm = (y - cy) / fy;
    
    // Calculate radial distance
    float r2 = x_norm * x_norm + y_norm * y_norm;
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    
    // Radial distortion correction
    float radial_factor = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
    
    // Tangential distortion correction
    float x_corrected = x_norm * radial_factor + 2.0f * p1 * x_norm * y_norm + p2 * (r2 + 2.0f * x_norm * x_norm);
    float y_corrected = y_norm * radial_factor + p1 * (r2 + 2.0f * y_norm * y_norm) + 2.0f * p2 * x_norm * y_norm;
    
    // Convert back to pixel coordinates
    float src_x = x_corrected * fx + cx;
    float src_y = y_corrected * fy + cy;
    
    // Bilinear interpolation for 3-channel image
    int dst_idx = (y * width + x) * 3;
    
    if (src_x >= 0 && src_x < width - 1 && src_y >= 0 && src_y < height - 1) {
        for (int c = 0; c < 3; ++c) {
            d_output[dst_idx + c] = static_cast<uint8_t>(
                bilinearInterpolation(d_input, src_x, src_y, width, height, 3, c));
        }
    } else {
        // Set to black if outside bounds
        d_output[dst_idx] = 0;
        d_output[dst_idx + 1] = 0;
        d_output[dst_idx + 2] = 0;
    }
}

__global__ void bgrToGrayscaleKernel(
    const uint8_t* d_input, uint8_t* d_output,
    int width, int height, float sigma) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int input_idx = (y * width + x) * 3;
    int output_idx = y * width + x;
    
    // BGR to grayscale conversion (OpenCV weights)
    float gray = 0.114f * d_input[input_idx] +     // Blue
                 0.587f * d_input[input_idx + 1] + // Green  
                 0.299f * d_input[input_idx + 2];  // Red
    
    d_output[output_idx] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, gray)));
}

// Feature detection kernels
__global__ void harrisCornerDetectionKernel(
    const uint8_t* d_image, float* d_corners, float2* d_keypoints,
    int* d_keypoint_count, int width, int height, float threshold,
    int max_keypoints) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2) return;
    
    int idx = y * width + x;
    
    // Calculate gradients using Sobel operators
    float Ix = 0, Iy = 0;
    
    // Sobel X gradient
    Ix += -1 * d_image[(y-1) * width + (x-1)];
    Ix += -2 * d_image[y * width + (x-1)];
    Ix += -1 * d_image[(y+1) * width + (x-1)];
    Ix += 1 * d_image[(y-1) * width + (x+1)];
    Ix += 2 * d_image[y * width + (x+1)];
    Ix += 1 * d_image[(y+1) * width + (x+1)];
    
    // Sobel Y gradient
    Iy += -1 * d_image[(y-1) * width + (x-1)];
    Iy += -2 * d_image[(y-1) * width + x];
    Iy += -1 * d_image[(y-1) * width + (x+1)];
    Iy += 1 * d_image[(y+1) * width + (x-1)];
    Iy += 2 * d_image[(y+1) * width + x];
    Iy += 1 * d_image[(y+1) * width + (x+1)];
    
    // Structure tensor components
    float Ixx = Ix * Ix;
    float Iyy = Iy * Iy;
    float Ixy = Ix * Iy;
    
    // Harris corner response
    float k = 0.04f;
    float det = Ixx * Iyy - Ixy * Ixy;
    float trace = Ixx + Iyy;
    float harris_response = det - k * trace * trace;
    
    d_corners[idx] = harris_response;
    
    // Non-maximum suppression and keypoint extraction
    if (harris_response > threshold) {
        bool is_local_max = true;
        
        // Check 3x3 neighborhood
        for (int dy = -1; dy <= 1 && is_local_max; ++dy) {
            for (int dx = -1; dx <= 1 && is_local_max; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int neighbor_idx = (y + dy) * width + (x + dx);
                if (d_corners[neighbor_idx] > harris_response) {
                    is_local_max = false;
                }
            }
        }
        
        if (is_local_max) {
            int keypoint_idx = atomicAdd(d_keypoint_count, 1);
            if (keypoint_idx < max_keypoints) {
                d_keypoints[keypoint_idx] = make_float2(x, y);
            }
        }
    }
}

__global__ void orbFeatureDetectionKernel(
    const uint8_t* d_image, float2* d_keypoints, uint8_t* d_descriptors,
    int* d_keypoint_count, int width, int height, int max_features,
    float scale_factor, int n_levels, int edge_threshold) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < edge_threshold || x >= width - edge_threshold || 
        y < edge_threshold || y >= height - edge_threshold) return;
    
    // FAST corner detection (simplified)
    int idx = y * width + x;
    uint8_t center_intensity = d_image[idx];
    int threshold = 50; // FAST threshold
    
    // Check FAST circle pattern (16 points)
    int circle_offsets[16][2] = {
        {0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
        {0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}
    };
    
    int consecutive_brighter = 0;
    int consecutive_darker = 0;
    int max_consecutive = 0;
    
    for (int i = 0; i < 16; ++i) {
        int px = x + circle_offsets[i][0];
        int py = y + circle_offsets[i][1];
        
        if (px >= 0 && px < width && py >= 0 && py < height) {
            uint8_t pixel_intensity = d_image[py * width + px];
            
            if (pixel_intensity > center_intensity + threshold) {
                consecutive_brighter++;
                consecutive_darker = 0;
            } else if (pixel_intensity < center_intensity - threshold) {
                consecutive_darker++;
                consecutive_brighter = 0;
            } else {
                consecutive_brighter = 0;
                consecutive_darker = 0;
            }
            
            max_consecutive = max(max_consecutive, max(consecutive_brighter, consecutive_darker));
        }
    }
    
    // If we have 9 or more consecutive pixels above/below threshold, it's a corner
    if (max_consecutive >= 9) {
        int keypoint_idx = atomicAdd(d_keypoint_count, 1);
        if (keypoint_idx < max_features) {
            d_keypoints[keypoint_idx] = make_float2(x, y);
            
            // Simplified ORB descriptor computation
            uint8_t* descriptor = &d_descriptors[keypoint_idx * ORB_DESCRIPTOR_SIZE];
            
            // Initialize descriptor to zero
            for (int i = 0; i < ORB_DESCRIPTOR_SIZE; ++i) {
                descriptor[i] = 0;
            }
            
            // Simplified binary pattern (production version would use rotated BRIEF)
            int pattern_size = 8;
            for (int i = 0; i < ORB_DESCRIPTOR_SIZE * 8 && i < pattern_size * pattern_size; ++i) {
                int bit_idx = i % 8;
                int byte_idx = i / 8;
                
                int dx1 = (i % pattern_size) - pattern_size/2;
                int dy1 = (i / pattern_size) - pattern_size/2;
                int dx2 = ((i + pattern_size/2) % pattern_size) - pattern_size/2;
                int dy2 = ((i + pattern_size/2) / pattern_size) - pattern_size/2;
                
                int px1 = x + dx1;
                int py1 = y + dy1;
                int px2 = x + dx2;
                int py2 = y + dy2;
                
                if (px1 >= 0 && px1 < width && py1 >= 0 && py1 < height &&
                    px2 >= 0 && px2 < width && py2 >= 0 && py2 < height) {
                    
                    uint8_t val1 = d_image[py1 * width + px1];
                    uint8_t val2 = d_image[py2 * width + px2];
                    
                    if (val1 > val2) {
                        descriptor[byte_idx] |= (1 << bit_idx);
                    }
                }
            }
        }
    }
}

// Feature matching kernels
__global__ void bruteForceMatchingKernel(
    const uint8_t* d_descriptors1, const uint8_t* d_descriptors2,
    int2* d_matches, float* d_distances, int* d_match_count,
    int desc_count1, int desc_count2, int desc_length,
    float ratio_threshold) {
    
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx >= desc_count1) return;
    
    const uint8_t* query_desc = &d_descriptors1[query_idx * desc_length];
    
    int best_match_idx = -1;
    int second_best_match_idx = -1;
    int best_distance = INT_MAX;
    int second_best_distance = INT_MAX;
    
    // Find best and second-best matches
    for (int train_idx = 0; train_idx < desc_count2; ++train_idx) {
        const uint8_t* train_desc = &d_descriptors2[train_idx * desc_length];
        
        int distance = hammingDistance(query_desc, train_desc, desc_length);
        
        if (distance < best_distance) {
            second_best_distance = best_distance;
            second_best_match_idx = best_match_idx;
            best_distance = distance;
            best_match_idx = train_idx;
        } else if (distance < second_best_distance) {
            second_best_distance = distance;
            second_best_match_idx = train_idx;
        }
    }
    
    // Apply Lowe's ratio test
    if (best_match_idx >= 0 && second_best_distance > 0) {
        float ratio = static_cast<float>(best_distance) / static_cast<float>(second_best_distance);
        
        if (ratio < ratio_threshold) {
            int match_idx = atomicAdd(d_match_count, 1);
            d_matches[match_idx] = make_int2(query_idx, best_match_idx);
            d_distances[match_idx] = static_cast<float>(best_distance);
        }
    }
}

// Image warping kernels
__global__ void perspectiveWarpKernel(
    const uint8_t* d_input, uint8_t* d_output,
    const float* d_homography, int input_width, int input_height,
    int output_width, int output_height, int channels,
    int interpolation_mode) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= output_width || y >= output_height) return;
    
    // Apply inverse homography to find source coordinates
    float src_x, src_y;
    applyHomographyTransform(static_cast<float>(x), static_cast<float>(y), d_homography, src_x, src_y);
    
    int output_idx = (y * output_width + x) * channels;
    
    // Check bounds
    if (src_x >= 0 && src_x < input_width - 1 && src_y >= 0 && src_y < input_height - 1) {
        for (int c = 0; c < channels; ++c) {
            if (interpolation_mode == 1) { // Linear interpolation
                d_output[output_idx + c] = static_cast<uint8_t>(
                    bilinearInterpolation(d_input, src_x, src_y, input_width, input_height, channels, c));
            } else { // Nearest neighbor
                int src_x_int = __float2int_rn(src_x);
                int src_y_int = __float2int_rn(src_y);
                int input_idx = (src_y_int * input_width + src_x_int) * channels;
                d_output[output_idx + c] = d_input[input_idx + c];
            }
        }
    } else {
        // Set to black for out-of-bounds
        for (int c = 0; c < channels; ++c) {
            d_output[output_idx + c] = 0;
        }
    }
}

__global__ void generateWarpMapKernel(
    float* d_warp_map_x, float* d_warp_map_y,
    const float* d_homography, int width, int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    float src_x, src_y;
    applyHomographyTransform(static_cast<float>(x), static_cast<float>(y), d_homography, src_x, src_y);
    
    d_warp_map_x[idx] = src_x;
    d_warp_map_y[idx] = src_y;
}

// Image blending kernels
__global__ void linearBlendingKernel(
    const uint8_t* d_img1, const uint8_t* d_img2,
    const float* d_weights, uint8_t* d_result,
    int width, int height, int channels) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int pixel_idx = idx * channels;
    
    float weight = d_weights[idx];
    float inv_weight = 1.0f - weight;
    
    for (int c = 0; c < channels; ++c) {
        float blended = weight * d_img1[pixel_idx + c] + inv_weight * d_img2[pixel_idx + c];
        d_result[pixel_idx + c] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, blended)));
    }
}

__global__ void multibandBlendingKernel(
    const uint8_t* d_img1, const uint8_t* d_img2,
    const float* d_weights1, const float* d_weights2,
    uint8_t* d_result, int width, int height, int channels,
    int num_bands, float blend_strength) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int pixel_idx = idx * channels;
    
    float weight1 = d_weights1[idx];
    float weight2 = d_weights2[idx];
    float total_weight = weight1 + weight2;
    
    if (total_weight > 0.0f) {
        weight1 /= total_weight;
        weight2 /= total_weight;
        
        for (int c = 0; c < channels; ++c) {
            float blended = weight1 * d_img1[pixel_idx + c] + weight2 * d_img2[pixel_idx + c];
            d_result[pixel_idx + c] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, blended)));
        }
    } else {
        for (int c = 0; c < channels; ++c) {
            d_result[pixel_idx + c] = 0;
        }
    }
}

// Utility kernels
__global__ void homographyRansacKernel(
    const float2* d_src_points, const float2* d_dst_points,
    float* d_homography, uchar* d_inlier_mask,
    int point_count, float threshold, float confidence,
    int max_iterations, int* d_inlier_count) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= max_iterations) return;
    
    // Initialize random state
    curandState state;
    curand_init(tid, 0, 0, &state);
    
    // Randomly select 4 points for homography estimation
    int selected_indices[4];
    for (int i = 0; i < 4; ++i) {
        selected_indices[i] = curand(&state) % point_count;
    }
    
    // Create local homography matrix
    float local_homography[9];
    
    // Simplified DLT homography estimation (production would use more robust method)
    // For brevity, using simplified approach - in production, implement full DLT
    
    // Set identity for now (placeholder)
    for (int i = 0; i < 9; ++i) {
        local_homography[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    }
    
    // Count inliers
    int local_inlier_count = 0;
    for (int i = 0; i < point_count; ++i) {
        float2 src = d_src_points[i];
        float2 dst = d_dst_points[i];
        
        float transformed_x, transformed_y;
        applyHomographyTransform(src.x, src.y, local_homography, transformed_x, transformed_y);
        
        float error = sqrtf((transformed_x - dst.x) * (transformed_x - dst.x) + 
                          (transformed_y - dst.y) * (transformed_y - dst.y));
        
        if (error < threshold) {
            local_inlier_count++;
        }
    }
    
    // Use atomic operations to update best result if this iteration is better
    int current_best = atomicMax(d_inlier_count, local_inlier_count);
    if (local_inlier_count > current_best) {
        // Copy homography to global memory
        for (int i = 0; i < 9; ++i) {
            d_homography[i] = local_homography[i];
        }
        
        // Update inlier mask
        for (int i = 0; i < point_count; ++i) {
            float2 src = d_src_points[i];
            float2 dst = d_dst_points[i];
            
            float transformed_x, transformed_y;
            applyHomographyTransform(src.x, src.y, local_homography, transformed_x, transformed_y);
            
            float error = sqrtf((transformed_x - dst.x) * (transformed_x - dst.x) + 
                              (transformed_y - dst.y) * (transformed_y - dst.y));
            
            d_inlier_mask[i] = (error < threshold) ? 1 : 0;
        }
    }
}

__global__ void calculateReprojectionErrorKernel(
    const float2* d_src_points, const float2* d_dst_points,
    const float* d_homography, float* d_errors,
    int point_count) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= point_count) return;
    
    float2 src = d_src_points[idx];
    float2 dst = d_dst_points[idx];
    
    float transformed_x, transformed_y;
    applyHomographyTransform(src.x, src.y, d_homography, transformed_x, transformed_y);
    
    float error = sqrtf((transformed_x - dst.x) * (transformed_x - dst.x) + 
                       (transformed_y - dst.y) * (transformed_y - dst.y));
    
    d_errors[idx] = error;
}

} // namespace camera_stitching

// C-style wrapper functions for external linkage
extern "C" {

cudaError_t launchUndistortImageKernel(
    const uint8_t* d_input, uint8_t* d_output,
    const float* d_camera_matrix, const float* d_dist_coeffs,
    int width, int height, cudaStream_t stream) {
    
    dim3 block_size(BLOCK_SIZE_16, BLOCK_SIZE_16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    camera_stitching::undistortImageKernel<<<grid_size, block_size, 0, stream>>>(
        d_input, d_output, d_camera_matrix, d_dist_coeffs, width, height);
    
    return cudaGetLastError();
}

cudaError_t launchBgrToGrayscaleKernel(
    const uint8_t* d_input, uint8_t* d_output,
    int width, int height, float sigma, cudaStream_t stream) {
    
    dim3 block_size(BLOCK_SIZE_16, BLOCK_SIZE_16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    camera_stitching::bgrToGrayscaleKernel<<<grid_size, block_size, 0, stream>>>(
        d_input, d_output, width, height, sigma);
    
    return cudaGetLastError();
}

cudaError_t launchOrbFeatureDetectionKernel(
    const uint8_t* d_image, float2* d_keypoints, uint8_t* d_descriptors,
    int* d_keypoint_count, int width, int height, int max_features,
    float scale_factor, int n_levels, int edge_threshold, cudaStream_t stream) {
    
    dim3 block_size(BLOCK_SIZE_16, BLOCK_SIZE_16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    // Initialize keypoint count to zero
    cudaMemsetAsync(d_keypoint_count, 0, sizeof(int), stream);
    
    camera_stitching::orbFeatureDetectionKernel<<<grid_size, block_size, 0, stream>>>(
        d_image, d_keypoints, d_descriptors, d_keypoint_count,
        width, height, max_features, scale_factor, n_levels, edge_threshold);
    
    return cudaGetLastError();
}

cudaError_t launchHarrisCornerDetectionKernel(
    const uint8_t* d_image, float* d_corners, float2* d_keypoints,
    int* d_keypoint_count, int width, int height, float threshold,
    int max_keypoints, cudaStream_t stream) {
    
    dim3 block_size(BLOCK_SIZE_16, BLOCK_SIZE_16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    // Initialize keypoint count to zero
    cudaMemsetAsync(d_keypoint_count, 0, sizeof(int), stream);
    
    camera_stitching::harrisCornerDetectionKernel<<<grid_size, block_size, 0, stream>>>(
        d_image, d_corners, d_keypoints, d_keypoint_count,
        width, height, threshold, max_keypoints);
    
    return cudaGetLastError();
}

cudaError_t launchBruteForceMatchingKernel(
    const uint8_t* d_descriptors1, const uint8_t* d_descriptors2,
    int2* d_matches, float* d_distances, int* d_match_count,
    int desc_count1, int desc_count2, int desc_length,
    float ratio_threshold, cudaStream_t stream) {
    
    int block_size = 256;
    int grid_size = (desc_count1 + block_size - 1) / block_size;
    
    // Initialize match count to zero
    cudaMemsetAsync(d_match_count, 0, sizeof(int), stream);
    
    camera_stitching::bruteForceMatchingKernel<<<grid_size, block_size, 0, stream>>>(
        d_descriptors1, d_descriptors2, d_matches, d_distances, d_match_count,
        desc_count1, desc_count2, desc_length, ratio_threshold);
    
    return cudaGetLastError();
}

cudaError_t launchPerspectiveWarpKernel(
    const uint8_t* d_input, uint8_t* d_output,
    const float* d_homography, int input_width, int input_height,
    int output_width, int output_height, int channels,
    int interpolation_mode, cudaStream_t stream) {
    
    dim3 block_size(BLOCK_SIZE_16, BLOCK_SIZE_16);
    dim3 grid_size((output_width + block_size.x - 1) / block_size.x,
                   (output_height + block_size.y - 1) / block_size.y);
    
    camera_stitching::perspectiveWarpKernel<<<grid_size, block_size, 0, stream>>>(
        d_input, d_output, d_homography, input_width, input_height,
        output_width, output_height, channels, interpolation_mode);
    
    return cudaGetLastError();
}

cudaError_t launchGenerateWarpMapKernel(
    float* d_warp_map_x, float* d_warp_map_y,
    const float* d_homography, int width, int height, cudaStream_t stream) {
    
    dim3 block_size(BLOCK_SIZE_16, BLOCK_SIZE_16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    camera_stitching::generateWarpMapKernel<<<grid_size, block_size, 0, stream>>>(
        d_warp_map_x, d_warp_map_y, d_homography, width, height);
    
    return cudaGetLastError();
}

cudaError_t launchMultibandBlendingKernel(
    const uint8_t* d_img1, const uint8_t* d_img2,
    const float* d_weights1, const float* d_weights2,
    uint8_t* d_result, int width, int height, int channels,
    int num_bands, float blend_strength, cudaStream_t stream) {
    
    dim3 block_size(BLOCK_SIZE_16, BLOCK_SIZE_16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    camera_stitching::multibandBlendingKernel<<<grid_size, block_size, 0, stream>>>(
        d_img1, d_img2, d_weights1, d_weights2, d_result,
        width, height, channels, num_bands, blend_strength);
    
    return cudaGetLastError();
}

cudaError_t launchLinearBlendingKernel(
    const uint8_t* d_img1, const uint8_t* d_img2,
    const float* d_weights, uint8_t* d_result,
    int width, int height, int channels, cudaStream_t stream) {
    
    dim3 block_size(BLOCK_SIZE_16, BLOCK_SIZE_16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    camera_stitching::linearBlendingKernel<<<grid_size, block_size, 0, stream>>>(
        d_img1, d_img2, d_weights, d_result, width, height, channels);
    
    return cudaGetLastError();
}

cudaError_t launchHomographyRansacKernel(
    const float2* d_src_points, const float2* d_dst_points,
    float* d_homography, uchar* d_inlier_mask,
    int point_count, float threshold, float confidence,
    int max_iterations, int* d_inlier_count, cudaStream_t stream) {
    
    int block_size = 256;
    int grid_size = (max_iterations + block_size - 1) / block_size;
    
    // Initialize inlier count to zero
    cudaMemsetAsync(d_inlier_count, 0, sizeof(int), stream);
    
    camera_stitching::homographyRansacKernel<<<grid_size, block_size, 0, stream>>>(
        d_src_points, d_dst_points, d_homography, d_inlier_mask,
        point_count, threshold, confidence, max_iterations, d_inlier_count);
    
    return cudaGetLastError();
}

cudaError_t launchCalculateReprojectionErrorKernel(
    const float2* d_src_points, const float2* d_dst_points,
    const float* d_homography, float* d_errors,
    int point_count, cudaStream_t stream) {
    
    int block_size = 256;
    int grid_size = (point_count + block_size - 1) / block_size;
    
    camera_stitching::calculateReprojectionErrorKernel<<<grid_size, block_size, 0, stream>>>(
        d_src_points, d_dst_points, d_homography, d_errors, point_count);
    
    return cudaGetLastError();
}

} // extern "C"