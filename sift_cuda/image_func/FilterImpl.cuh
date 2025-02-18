#pragma once

namespace sift_cuda {

    template <ImageContainerType Input_T, ImageContainerType Output_T, ImageContainerType Filter_T, ImageContainerType Temp_T>
    void applyFilter(
        CParamType_t<Input_T> input,
        MParamType_t<Output_T> output,
        CParamType_t<Filter_T> filter,
        CParamType_t<Temp_T> temp,
        cudaStream_t stream
    ) {
        // if (!output.isAllocated()) {
        //     output.allocate(input.getDim().x, input.getDim().y);
        // }
        
        dim3 blockDim; dim3 gridDim;
        calculateGridConfig(input.getDim(), gridDim, blockDim);
        
        int filter_radius = filter.getDim().x / 2; // 10

        // Very important to call vertical first
        apply1DFilterToPixel<<<gridDim, blockDim, 0, stream>>>(
            input.getData(),
            temp.getData(),
            filter.getData(),
            input.getDim(),
            filter.getDim(),
            input.getPitch(),
            temp.getPitch(),
            filter.getPitch(),
            true
        );

        apply1DFilterToPixel<<<gridDim, blockDim, 0, stream>>>(
            temp.getData(),
            output.getData(),
            filter.getData(),
            input.getDim(),
            filter.getDim(),
            temp.getPitch(),
            output.getPitch(),
            filter.getPitch(),
            false
        );
    }
}