// lane_detection/cuda/tensor_kernels.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

namespace lane_detection {

// CUDA kernel for image preprocessing (BGR to RGB + normalization)
__global__ void preprocessImageKernel(
    const uint8_t* input,
    float* output,
    int width, int height,
    const float* mean,
    const float* std,
    float scale) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx >= total_pixels) return;

    int y = idx / width;
    int x = idx % width;

    // Input is BGR format, convert to RGB and normalize
    int bgr_idx = idx * 3;

    // BGR to RGB conversion and normalization
    float b = static_cast<float>(input[bgr_idx + 0]) / scale;
    float g = static_cast<float>(input[bgr_idx + 1]) / scale;
    float r = static_cast<float>(input[bgr_idx + 2]) / scale;

    // Normalize using ImageNet statistics
    float r_norm = (r - mean[0]) / std[0];
    float g_norm = (g - mean[1]) / std[1];
    float b_norm = (b - mean[2]) / std[2];

    // Output in CHW format (Channel-Height-Width)
    int chw_r_idx = 0 * total_pixels + idx;
    int chw_g_idx = 1 * total_pixels + idx;
    int chw_b_idx = 2 * total_pixels + idx;

    output[chw_r_idx] = r_norm;
    output[chw_g_idx] = g_norm;
    output[chw_b_idx] = b_norm;
}

// CUDA kernel for bilinear interpolation resize
__global__ void resizeImageKernel(
    const float* input,
    float* output,
    int input_width, int input_height,
    int output_width, int output_height,
    int channels) {

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (out_x >= output_width || out_y >= output_height || c >= channels) return;

    // Calculate input coordinates
    float scale_x = static_cast<float>(input_width) / output_width;
    float scale_y = static_cast<float>(input_height) / output_height;

    float in_x = (out_x + 0.5f) * scale_x - 0.5f;
    float in_y = (out_y + 0.5f) * scale_y - 0.5f;

    // Clamp to input bounds
    in_x = fmaxf(0.0f, fminf(in_x, input_width - 1.0f));
    in_y = fmaxf(0.0f, fminf(in_y, input_height - 1.0f));

    // Bilinear interpolation
    int x0 = static_cast<int>(floorf(in_x));
    int y0 = static_cast<int>(floorf(in_y));
    int x1 = min(x0 + 1, input_width - 1);
    int y1 = min(y0 + 1, input_height - 1);

    float dx = in_x - x0;
    float dy = in_y - y0;

    int input_size = input_width * input_height;
    int output_size = output_width * output_height;

    int in_idx_00 = c * input_size + y0 * input_width + x0;
    int in_idx_01 = c * input_size + y0 * input_width + x1;
    int in_idx_10 = c * input_size + y1 * input_width + x0;
    int in_idx_11 = c * input_size + y1 * input_width + x1;

    float val_00 = input[in_idx_00];
    float val_01 = input[in_idx_01];
    float val_10 = input[in_idx_10];
    float val_11 = input[in_idx_11];

    float val_0 = val_00 * (1.0f - dx) + val_01 * dx;
    float val_1 = val_10 * (1.0f - dx) + val_11 * dx;
    float result = val_0 * (1.0f - dy) + val_1 * dy;

    int out_idx = c * output_size + out_y * output_width + out_x;
    output[out_idx] = result;
}

// CUDA kernel for post-processing segmentation masks
__global__ void processMaskKernel(
    const float* input_mask,
    uint8_t* output_mask,
    int width, int height,
    float threshold) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx >= total_pixels) return;

    float mask_value = input_mask[idx];
    output_mask[idx] = (mask_value > threshold) ? 255 : 0;
}

// CUDA kernel for NMS (Non-Maximum Suppression) support
__global__ void computeIoUKernel(
    const float* boxes,
    float* iou_matrix,
    int num_boxes) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= num_boxes || j >= num_boxes) return;

    if (i == j) {
        iou_matrix[i * num_boxes + j] = 1.0f;
        return;
    }

    // Box format: [x1, y1, x2, y2]
    float x1_i = boxes[i * 4 + 0];
    float y1_i = boxes[i * 4 + 1];
    float x2_i = boxes[i * 4 + 2];
    float y2_i = boxes[i * 4 + 3];

    float x1_j = boxes[j * 4 + 0];
    float y1_j = boxes[j * 4 + 1];
    float x2_j = boxes[j * 4 + 2];
    float y2_j = boxes[j * 4 + 3];

    // Compute intersection
    float inter_x1 = fmaxf(x1_i, x1_j);
    float inter_y1 = fmaxf(y1_i, y1_j);
    float inter_x2 = fminf(x2_i, x2_j);
    float inter_y2 = fminf(y2_i, y2_j);

    float inter_area = 0.0f;
    if (inter_x2 > inter_x1 && inter_y2 > inter_y1) {
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
    }

    // Compute union
    float area_i = (x2_i - x1_i) * (y2_i - y1_i);
    float area_j = (x2_j - x1_j) * (y2_j - y1_j);
    float union_area = area_i + area_j - inter_area;

    // Compute IoU
    float iou = (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
    iou_matrix[i * num_boxes + j] = iou;
}

} // namespace lane_detection

// C-style wrapper functions for CUDA kernel launches
extern "C" {

cudaError_t launchPreprocessKernel(
    const uint8_t* input,
    float* output,
    int width, int height,
    const float* mean,
    const float* std,
    float scale,
    cudaStream_t stream) {

    int total_pixels = width * height;
    int block_size = 256;
    int grid_size = (total_pixels + block_size - 1) / block_size;

    lane_detection::preprocessImageKernel<<<grid_size, block_size, 0, stream>>>(
        input, output, width, height, mean, std, scale);

    return cudaGetLastError();
}

cudaError_t launchResizeKernel(
    const float* input,
    float* output,
    int input_width, int input_height,
    int output_width, int output_height,
    int channels,
    cudaStream_t stream) {

    dim3 block_size(16, 16, 1);
    dim3 grid_size(
        (output_width + block_size.x - 1) / block_size.x,
        (output_height + block_size.y - 1) / block_size.y,
        channels
    );

    lane_detection::resizeImageKernel<<<grid_size, block_size, 0, stream>>>(
        input, output, input_width, input_height,
        output_width, output_height, channels);

    return cudaGetLastError();
}

cudaError_t launchMaskProcessKernel(
    const float* input_mask,
    uint8_t* output_mask,
    int width, int height,
    float threshold,
    cudaStream_t stream) {

    int total_pixels = width * height;
    int block_size = 256;
    int grid_size = (total_pixels + block_size - 1) / block_size;

    lane_detection::processMaskKernel<<<grid_size, block_size, 0, stream>>>(
        input_mask, output_mask, width, height, threshold);

    return cudaGetLastError();
}

cudaError_t launchIoUKernel(
    const float* boxes,
    float* iou_matrix,
    int num_boxes,
    cudaStream_t stream) {

    dim3 block_size(16, 16);
    dim3 grid_size(
        (num_boxes + block_size.x - 1) / block_size.x,
        (num_boxes + block_size.y - 1) / block_size.y
    );

    lane_detection::computeIoUKernel<<<grid_size, block_size, 0, stream>>>(
        boxes, iou_matrix, num_boxes);

    return cudaGetLastError();
}

} // extern "C"