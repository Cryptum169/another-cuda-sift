#include "MatOps.cuh"

#include "utils.cuh"

#include <numeric>
#include <thrust/device_vector.h>
#include <cub/device/device_scan.cuh>

namespace sift_cuda {
    __global__ void minus(
        const float* left,
        const float* right,
        float* result,
        const float2 size,
        size_t pitch,
        int px_per_thread
    ) {
        pitch = pitch / sizeof(float);  // convert to elements
        int x = (blockIdx.x * blockDim.x + threadIdx.x) * px_per_thread;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (y >= size.y) {
            return;
        }
        int y_base = y * pitch;

        #pragma unroll
        for (int px = 0; px < px_per_thread; px++) {
            int px_x = px + x;
            if (px_x >= size.x) {
                return;
            }
            
            int index = px_x + y_base;
            result[index] = left[index] - right[index];
        }
    }

    void findPeaks3D(
        const ImageBlock& imageBlock,
        float threshold,
        int border,
        KeypointCollections& result,
        cudaStream_t stream
    ) {
        dim3 block(16, 16);
        dim3 grid(
            (imageBlock.getDim().x + block.x - 1) / block.z,
            (imageBlock.getDim().y + block.y - 1) / block.y,
             imageBlock.getDim().z - 2
        );

        findPeaksBlockFirstPass<<<grid, block, 0, stream>>>(
            imageBlock.getData(),
            thrust::raw_pointer_cast(result.mask.data()),
            imageBlock.getDim(),
            imageBlock.getPitch(),
            threshold,
            border
        );

        // temporary memory not initialized yet
        // TODO: this is probably fixed size, move it to allocation phase
        if (result.temp_storage_bytes == 0) {
            cub::DeviceScan::InclusiveSum(
                nullptr, result.temp_storage_bytes,
                thrust::raw_pointer_cast(result.mask.data()),
                thrust::raw_pointer_cast(result.prefix_sum.data()),
                result.mask.size(), stream
            );
            result.temp_storage.resize(result.temp_storage_bytes);
        }

        cub::DeviceScan::InclusiveSum(
            thrust::raw_pointer_cast(result.temp_storage.data()),
            result.temp_storage_bytes,
            thrust::raw_pointer_cast(result.mask.data()),
            thrust::raw_pointer_cast(result.prefix_sum.data()),
            result.mask.size(), stream
        );

        findPeaksBlockSecondPass<<<grid, block, 0, stream>>>(
            thrust::raw_pointer_cast(result.mask.data()),
            thrust::raw_pointer_cast(result.prefix_sum.data()),
            thrust::raw_pointer_cast(result.candidateKpts.data()),
            imageBlock.getDim(),
            imageBlock.getPitch(),
            result.candidateKpts.size()
        );
    }

    __global__ void findPeaksBlockFirstPass(
        const float* block,
        int* buffer,
        const float3 size,
        size_t pitch,
        float threshold,
        int border
    ) {
        int center_x = threadIdx.x + blockDim.x * blockIdx.x;
        int center_y = threadIdx.y + blockDim.y * blockIdx.y;
        int img = blockIdx.z;

        if (img > (size.z - 3)) {
            return;
        }

        if (center_x >= (size.x - border) || center_y >= (size.y - border)) {
            return;
        }

        if (center_x < border || center_y < border) {
            return;
        }

        int h_dim = pitch / sizeof(float);

        const float* prev = block + (size_t)(h_dim * size.y * img);
        const float* curr = block + (size_t)(h_dim * size.y * (img + 1));
        const float* next = block + (size_t)(h_dim * size.y * (img + 2));

        int idx = center_y * h_dim + center_x;
        float center_val = curr[idx];
        bool maxima{false};
        if (abs(center_val) > threshold) {
            maxima = true;
            for (int y_idx = center_y - 1; y_idx < center_y + 2; y_idx++) {
                for (int x_idx = center_x - 1; x_idx < center_x + 2; x_idx++) {
                    if (center_val > 0) {
                        maxima &= (center_val >= prev[y_idx * h_dim + x_idx]);
                        maxima &= (center_val >= curr[y_idx * h_dim + x_idx]);
                        maxima &= (center_val >= next[y_idx * h_dim + x_idx]);
                    } else if (center_val < 0) {
                        maxima &= (center_val <= prev[y_idx * h_dim + x_idx]);
                        maxima &= (center_val <= curr[y_idx * h_dim + x_idx]);
                        maxima &= (center_val <= next[y_idx * h_dim + x_idx]);
                    } else {
                        maxima = false;
                        break;
                    }
                }
                if (!maxima) { break; }
            }
        }
        __threadfence();
        buffer[(size_t)(h_dim * size.y * img) + idx] = maxima ? 1 : 0;
    }

    __global__ void findPeaksBlockSecondPass(
        const int* mask,
        const int* prefix_sum,
        float3* kpts,
        const float3 size,
        size_t pitch,
        size_t max_num_kpts
    ) {
        int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
        int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
        int img = blockIdx.z;

        if (x_idx >= size.x || y_idx >= size.y) {
            return;
        }

        if (img >= (size.z - 2)) {
            return;
        }

        int img_offset = img * pitch * size.y / sizeof(float);
        int idx = x_idx + y_idx * pitch / sizeof(float) + img_offset;

        if (mask[idx] == 1) {
            int kpt_idx = prefix_sum[idx];

            if (kpt_idx >= max_num_kpts) {
                return;
            }

            kpts[kpt_idx - 1] = make_float3(x_idx, y_idx, img + 1);
        }
    }
}
