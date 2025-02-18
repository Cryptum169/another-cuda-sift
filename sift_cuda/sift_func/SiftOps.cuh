#pragma once

#include <cuda_fp16.h>

namespace sift_cuda {

    static const float SIFT_FIXPT_SCALE = 1.f;
    static const float SIFT_IMG_BORDER = 5.f;
    static const float SIFT_MAX_INTER_STEPS = 5.f;
    static const float SIFT_INIT_SIGMA = 0.5f;
    static const float SIFT_ORI_SIG_FCTR = 1.5f;
    static const float SIFT_DESCR_SCL_FCTR = 3.f;
    static const float SIFT_ORI_PEAK_RATIO = 0.8f;

    __device__ void solve_3x3_system(const float* A, const float* b, float* x);

    __global__ void adjustExtrema(
        float3* kpts,
        int* num_kpts,
        const int max_pts_this_octave,
        int octave,
        bool upscaled,
        const float* imageBlock,
        float4* features,
        int* mask,
        size_t pitch,
        float3 size,
        float edgeThreshold,
        float contrastThreshold,
        int numOctaveLayers,
        float sigma
    );

    __global__ void collectKpts(
        const int* mask,
        const int* prefix_sum,
        const float3* kpts,
        float3* output,
        const float4* features,
        float4* output_feature,
        size_t total_size,
        int* collect_size_ptr
    );

    __global__ void calOriHistMultiThread(
        const float* imageBlock,
        float3* kpts,
        float4* features,
        int* mask,
        const float2 size,
        const int pitch,
        int* total_num_kpts,
        const int max_expanded_kpts,
        const int oct_idx,
        const float radius_factor,
        const float peak_ratio,
        const float scale_factor,
        bool upscaled
    );

    // Can probably use cudaMemset instead.
    // might even be faster
    __global__ void memsetMask(
        int* mask,
        size_t size
    );

    __global__ void genDescriptorMultiThread(
        const float* imageBlock,
        const float3* kpt,
        const float4* feature,
        const int num_pts,
        const float scale_multiplier,
        const float3 block_size,
        const size_t pitch,
        half* descriptor
    );
}