#include "Match.cuh"

#include <stdio.h>
#include "sift_cuda/types/CudaMemRAII.cuh"

namespace sift_cuda {

    std::vector<int> matchBruteForce(
        const thrust::device_vector<half> & des,
        const int num_des, 
        const thrust::device_vector<half> & src,
        const int num_src
    ) {
        int block_size = 256;
        int grid_size = (num_des * 32 + block_size - 1) / block_size;
        thrust::device_vector<float> score(num_des * num_src, 0);
        thrust::device_vector<int> out_idx(num_des, -1);

        matchBruteForce<<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(des.data()),
            num_des, 
            thrust::raw_pointer_cast(src.data()),
            num_src,
            thrust::raw_pointer_cast(score.data()),
            thrust::raw_pointer_cast(out_idx.data())
        );

        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<int> stl_result(out_idx.size());
        thrust::copy(out_idx.begin(), out_idx.end(), stl_result.begin());

        return stl_result;
    }

    __global__ void matchBruteForce(
        const half* const des,
        const int num_des,
        const half* const src,
        const int num_src,
        float* dist_score,
        int* out_idx
    ) {
        constexpr int descriptor_dim = 128;
        constexpr int block_size = 256;
        constexpr int thread_per_des = 32;

        constexpr int src_batch = block_size / thread_per_des;
        __shared__ half this_des[block_size / thread_per_des * descriptor_dim];
        __shared__ half this_src[src_batch * descriptor_dim];

        int global_des_idx = threadIdx.x + blockDim.x * blockIdx.x;
        global_des_idx = global_des_idx / thread_per_des;
        int intra_block_idx = threadIdx.x / thread_per_des;
        int laneId = threadIdx.x % thread_per_des;

        if (global_des_idx < num_des) {
            half2* des_ptr = (half2*)(des + global_des_idx * descriptor_dim);
            half2* shared_des_ptr = (half2*)(this_des + descriptor_dim * intra_block_idx);
            
            #pragma unroll 4
            for (int idx = laneId; idx < descriptor_dim / 2; idx += thread_per_des) {
                shared_des_ptr[idx] = des_ptr[idx];
            }
        }

        // constexpr float multiplier = 4.0f;
        half2 inv = half2(__half2float(0.25f), __half2float(0.25f));
        // thread operates until the end of src
        for (int global_src_idx = 0; global_src_idx < num_src; global_src_idx += src_batch) {
            int current_batch_size = min(src_batch, num_src - global_src_idx);

            __syncthreads();

            if (intra_block_idx < current_batch_size) {
                half2* src_ptr = (half2*)(src + (global_src_idx + intra_block_idx) * descriptor_dim);
                half2* shared_src_ptr = (half2*)(this_src + descriptor_dim * intra_block_idx);
                
                for (int idx = laneId; idx < descriptor_dim / 2; idx += thread_per_des) {
                    shared_src_ptr[idx] = src_ptr[idx];
                }
            }
            __syncthreads();

            // To prevent overfloat (overflow really)
            // Creates discrepancies at 1 decimal place, final scores typically at ~2k. Acceptable numerical issue
            for (int shared_src_idx = 0; shared_src_idx < current_batch_size; shared_src_idx++) {
                float score = 0.f;
                if (global_des_idx >= num_des) {
                    break;
                }

                const half2* des_vec = (half2*)(this_des + descriptor_dim * intra_block_idx);
                const half2* src_vec = (half2*)(this_src + shared_src_idx * descriptor_dim);

                for (int idx = laneId; idx < descriptor_dim / 2; idx += thread_per_des) {
                    half2 diff = des_vec[idx] - src_vec[idx];
                    diff = diff * inv;
                    half2 sqr = diff * diff;
                    score += fmaf(1, __half2float(sqr.x), __half2float(sqr.y));
                }
                // I say this is not even needed
                // score = score * multiplier * multiplier;

                // Do a warp-level reduction first
                // Although it's thread_per_des / 2, since this is per-warp reduction, this will ever only work for thread_per_des = 32
                #pragma unroll
                for (int offset = thread_per_des / 2; offset > 0; offset /= 2) {
                    score += __shfl_down_sync(0xffffffff, score, offset);
                }

                if (laneId == 0) {
                    // Atomic is not needed here
                    dist_score[global_des_idx * num_src + global_src_idx + shared_src_idx] = score;
                }
            }
        }

        __syncthreads();

        if (global_des_idx >= num_des) {
            return;
        }

        // Hard coded Lowe's test
        float min_dist = 1e6;
        int min_idx = -1;

        float min_dist_2 = 1e6;
        int min_idx_2 = -1;

        int num_element_to_look_at = ceil(float(num_src) / thread_per_des);
        float* dist_offset = dist_score + global_des_idx * num_src;

        int ceiling = min(num_src, (laneId + 1) * num_element_to_look_at);
        // For each thread, find lowest 2 values and their indices
        for (int idx = laneId * num_element_to_look_at; idx < ceiling; idx++) {
            float val = dist_offset[idx];

            if (val < min_dist) {
                min_dist_2 = min_dist;
                min_idx_2 = min_idx;

                min_dist = val;
                min_idx = idx;
            } else if (val < min_dist_2) {
                min_dist_2 = val;
                min_idx_2 = idx;
            }
        }

        // Reduce across the warp
        for (int offset = thread_per_des / 2; offset > 0; offset /= 2) {
            float nearby_min = __shfl_down_sync(0xffffffff, min_dist, offset);
            float nearby_min_idx = __shfl_down_sync(0xffffffff, min_idx, offset);
            if (nearby_min < min_dist) {
                min_dist_2 = min_dist;
                min_idx_2 = min_idx;

                min_dist = nearby_min;
                min_idx = nearby_min_idx;
            }

            nearby_min = __shfl_down_sync(0xffffffff, min_dist_2, offset);
            nearby_min_idx = __shfl_down_sync(0xffffffff, min_idx_2, offset);
            if (nearby_min < min_dist_2) {
                min_dist_2 = nearby_min;
                min_idx_2 = nearby_min_idx;
            }
        }

        if (laneId == 0) {
            if (min_dist < 0.8 * min_dist_2) {
                out_idx[global_des_idx] = min_idx;
            }
        }

    }
}
