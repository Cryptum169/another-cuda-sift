#pragma once

#include <cuda_runtime.h>

namespace sift_cuda {

    __host__ void calculateGridConfig(
        const float2& imageDim,
        dim3& gridDim,
        dim3& blockDim,
        const unsigned int maxThreadsPerBlock = 256
    );
}
