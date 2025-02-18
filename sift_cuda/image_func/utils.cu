#include "utils.cuh"

namespace sift_cuda {
    __host__ void calculateGridConfig(
        const float2& imageDim,
        dim3& gridDim,
        dim3& blockDim,
        const unsigned int maxThreadsPerBlock
    ) {
        // 256 threads
        blockDim.x = 16;
        blockDim.y = 16;

        unsigned int blocksNeeded = (imageDim.x + blockDim.x - 1) / blockDim.x *
                                (imageDim.y + blockDim.y - 1) / blockDim.y;

        if (blocksNeeded < 4) {
            blockDim.x = 8;
            blockDim.y = 8;
        }

        // (num + blocksize - 1) / blocksize
        gridDim.x = (imageDim.x + blockDim.x - 1) / blockDim.x;
        gridDim.y = (imageDim.y + blockDim.y - 1) / blockDim.y;
    }
}
