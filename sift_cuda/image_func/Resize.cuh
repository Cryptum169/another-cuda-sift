#pragma once

#include "sift_cuda/types/CudaImage.cuh"
#include "utils.cuh"

namespace sift_cuda {

    template <ImageContainerType Input_T, ImageContainerType Output_T>
    void resize_cuda(
        const Input_T& input,
        Output_T& output,
        float2 target_size,
        cudaStream_t stream = nullptr
    );

    __global__ void resize_cuda_bilinear(
        const float* input,
        float* output,
        const float2 input_size,
        const float2 output_size,
        size_t input_pitch,
        size_t output_pitch
    );

    inline __device__ float2 coordQuery(const float2 input, const float2 output, const int2 query)  {
        float x = (query.x + 0.5) * input.x / output.x - 0.5;
        float y = (query.y + 0.5) * input.y / output.y - 0.5;
        return make_float2(x, y);
    }
};

#include "ResizeImpl.cuh"
