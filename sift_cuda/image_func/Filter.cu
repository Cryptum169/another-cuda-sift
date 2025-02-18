#include "Filter.cuh"

#include "utils.cuh"
#include <cassert>

namespace sift_cuda {

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
    ) {
        int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
        if (x_idx >= input_size.x || y_idx >= input_size.y) {
            return;
        }
        int filter_rad = filter_size.x / 2;

        input_pitch /= sizeof(float);
        output_pitch /= sizeof(float);
        filter_pitch /= sizeof(float);

        float value = 0.f;
        int index = x_idx + y_idx * output_pitch;
        if (!vertical) {
            for (int px_offset = -filter_rad; px_offset < filter_rad + 1; px_offset++) {
                int px = reflect101(x_idx + px_offset, input_size.x);

                float img_val = input[y_idx * input_pitch + px];
                float filter_val = filter[px_offset + filter_rad];
                value = __fmaf_rn(img_val, filter_val, value);
            }
            output[index] = value;
        } else {
            for (int py_offset = -filter_rad; py_offset < filter_rad + 1; py_offset++) {
                int py = reflect101(y_idx + py_offset, input_size.y);

                float img_val = input[py * input_pitch + x_idx];
                float filter_val = filter[py_offset + filter_rad];
                value = __fmaf_rn(img_val, filter_val, value);
            }
            output[index] = value;
        }
    }

}
