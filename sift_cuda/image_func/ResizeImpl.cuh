#pragma once

namespace sift_cuda {

    template <ImageContainerType Input_T, ImageContainerType Output_T>
    void resize_cuda(
        const Input_T& input,
        Output_T& output,
        float2 target_size,
        cudaStream_t stream
    ){
        // if (!output.isAllocated()) {
        //     output.allocate(target_size.x, target_size.y);
        // }

        dim3 blockDim; dim3 gridDim;
        calculateGridConfig(output.getDim(), gridDim, blockDim);

        resize_cuda_bilinear<<<gridDim, blockDim, 0, stream>>>(
            input.getData(),
            output.getData(),
            input.getDim(),
            output.getDim(),
            input.getPitch(),
            output.getPitch()
        );
        return;
    }
}
