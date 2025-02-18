#pragma once

#include "sift_cuda/types/CudaImage.cuh"
#include "utils.cuh"

namespace sift_cuda {
    enum FilterType {
        PITCHED_MEMORY,
        TEXTURE,
        DOUBLE_1D
    };

    __global__ void apply1DFilterToPixel(
        const float* input,
        float* output,
        const float* filter,
        const float2 input_size,
        const float2 filter_size,
        size_t input_pitch,
        size_t output_pitch,
        size_t filter_pitch,
        bool vertical
    );

    // Code for fast mod
    struct ModConstant {
        int divisor;
        int multiplier;
        int shift = 32;

        ModConstant(int div) {
            divisor = 2 * div;
            uint64_t n = (1ull << (shift + 32)) / divisor;

            if (((n * divisor) >> shift) != (1ull << 32))
            {
                n++;
            }

            multiplier = static_cast<uint32_t>(n);
        }
    };

    inline __device__ int fast_mod(int x, int3 mc) {
        // x is divisor
        // y is multiplier
        // z is shift (32)
        uint32_t q = ((uint64_t) mc.y * x) >> mc.z;
        return x - q * mc.x;
    }

    inline __device__ int reflect101(int idx, int length) {
        if (length <= 1) return 0;

        idx = idx < 0 ? (-idx) : idx;
        if (idx >= length) {
            int period = 2 * (length - 1);
            // TODO: some day use fast mod again
            idx = idx % period;
            if (idx >= length) {
                idx = period - idx;
            }
        }

        return idx;
    }

    template <ImageContainerType Input_T, ImageContainerType Output_T, ImageContainerType Filter_T, ImageContainerType Temp_T>
    void applyFilter(
        CParamType_t<Input_T> input,
        MParamType_t<Output_T> output,
        CParamType_t<Filter_T> filter,
        CParamType_t<Temp_T> temp,
        cudaStream_t stream = nullptr
    );
}

#include "FilterImpl.cuh"
