#pragma once

#include "sift_cuda/types/CudaImage.cuh"
#include "sift_cuda/types/KeyPoint.cuh"

#include "utils.cuh"

namespace sift_cuda {

    template <ImageContainerType Left_T, ImageContainerType Right_T, ImageContainerType Result_T>
    void minus(
        const Left_T& left,
        const Right_T& right,
        Result_T& result,
        cudaStream_t stream = nullptr
    );

    __global__ void minus(
        const float* left,
        const float* right,
        float* result,
        const float2 size,
        size_t pitch,
        int px_per_thread
    );

    void findPeaks3D(
        const ImageBlock& imageBlock,
        float threshold,
        int border,
        KeypointCollections& result,
        cudaStream_t stream = nullptr
    );

    __global__ void findPeaksBlockFirstPass(
        const float* block,
        int* buffer,
        const float3 size,
        size_t pitch,
        float threshold,
        int border
    );

    __global__ void findPeaksBlockSecondPass(
        const int* mask,
        const int* prefix_sum,
        float3* kpts,
        const float3 size,
        size_t pitch,
        size_t max_num_kpts
    );
};

#include "MatOpsImpl.cuh"
