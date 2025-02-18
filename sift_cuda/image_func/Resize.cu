#include "Resize.cuh"

#include "utils.cuh"

namespace sift_cuda {
    __global__ void resize_cuda_bilinear(
        const float* input,
        float* output,
        const float2 input_size,
        const float2 output_size,
        size_t input_pitch,
        size_t output_pitch
    ) {
        // size: x = col, y = row
        int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

        if (x_idx >= output_size.x) {
            return;
        }

        if (y_idx >= output_size.y) {
            return;
        }

        float coord_x = (x_idx + 0.5f) * input_size.x / output_size.x - 0.5f;
        float coord_y = (y_idx + 0.5f) * input_size.y / output_size.y - 0.5f;

        int max_x = int(input_size.x - 1);
        int max_y = int(input_size.y - 1);

        int x1 = __float2int_rd(coord_x); // floor
        x1 = (min(max_x, max(0, x1)));
        int y1 = __float2int_rd(coord_y);
        y1 = min(max_y, max(0, y1));

        float integral_x;
        float x_frac = modff(coord_x, &integral_x);
        int x2 = x_frac == 0 ? x1 + 1: __float2int_ru(coord_x);
        x2 = (min(max_x, max(0, x2)));

        float integral_y;
        float y_frac = modff(coord_y, &integral_y);
        int y2 = y_frac == 0 ? y1 + 1: __float2int_ru(coord_y);
        y2 = min(max_y, max(0, y2));

        float x2e = 1.f - x_frac;
        float x1e = x_frac;
        float y2e = 1.f - y_frac;
        float y1e = y_frac;
        
        float v11 = input[x1 + y1 * input_pitch / sizeof(float)];
        float v21 = input[x2 + y1 * input_pitch / sizeof(float)];
        float v12 = input[x1 + y2 * input_pitch / sizeof(float)];
        float v22 = input[x2 + y2 * input_pitch / sizeof(float)];

        float v21e = v21 * x1e;
        float v22e = v22 * x1e;
        float r11 = __fmaf_rn(v11, x2e, v21e) * y2e;
        float r12 = __fmaf_rn(v12, x2e, v22e) * y1e;

        int index = x_idx + y_idx * output_pitch / sizeof(float);
        output[index] = r11 + r12;
    }
}
