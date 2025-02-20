#pragma once

namespace sift_cuda {

    template <ImageContainerType Left_T, ImageContainerType Right_T, ImageContainerType Result_T>
    void minus(
        const Left_T& left,
        const Right_T& right,
        Result_T& result,
        cudaStream_t stream
    ) {
        dim3 blockDim(16,16);
        int px_per_thread{1};
        dim3 gridDim(
            (left.getDim().x + blockDim.x * px_per_thread - 1) / (blockDim.x * px_per_thread),
            (left.getDim().y + blockDim.y - 1) / blockDim.y
        );
        minus<<<gridDim, blockDim, 0, stream>>>(
            left.getData(),
            right.getData(),
            result.getData(),
            left.getDim(),
            left.getPitch(),
            px_per_thread
        );
    }
}
