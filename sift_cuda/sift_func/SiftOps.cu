#include "SiftOps.cuh"

namespace sift_cuda {

    // Faster than invoking CUBLAS
    __device__ void solve_3x3_system(const float* A, const float* b, float* x) {
        float a[3][3];
        float bb[3];
        float temp;

        #pragma unroll
        for (int i = 0; i < 3; i++) {
            bb[i] = b[i];
            #pragma unroll
            for (int j = 0; j < 3; j++) {
                a[i][j] = A[i * 3 + j];
            }
        }

        // Gaussian elimination with partial pivoting
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            int max_idx = k;
            float max_val = fabsf(a[k][k]);

            #pragma unroll
            for (int i = k + 1; i < 3; i++) {
                if (fabsf(a[i][k]) > max_val) {
                    max_val = fabsf(a[i][k]);
                    max_idx = i;
                }
            }

            if (max_idx != k) {
                #pragma unroll
                for (int j = k; j < 3; j++) {
                    temp = a[k][j];
                    a[k][j] = a[max_idx][j];
                    a[max_idx][j] = temp;
                }
                temp = bb[k];
                bb[k] = bb[max_idx];
                bb[max_idx] = temp;
            }

            #pragma unroll
            for (int i = k + 1; i < 3; i++) {
                temp = a[i][k] / a[k][k];
                bb[i] -= temp * bb[k];
                #pragma unroll
                for (int j = k; j < 3; j++) {
                    a[i][j] -= temp * a[k][j];
                }
            }
        }

        // Back substitution
        x[2] = bb[2] / a[2][2];
        x[1] = (bb[1] - a[1][2] * x[2]) / a[1][1];
        x[0] = (bb[0] - a[0][1] * x[1] - a[0][2] * x[2]) / a[0][0];
    }

    __global__ void adjustExtrema(
        float3* kpts,
        int* num_kpts,
        const int max_pts_this_octave,
        int octave,
        bool upscaled,
        const float* imageBlock,
        float4* features,
        int* mask,
        size_t pitch,
        float3 size,
        float edgeThreshold,
        float contrastThreshold,
        int numOctaveLayers,
        float sigma
    ) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        if (idx >= num_kpts[0] || idx >= max_pts_this_octave) {
            return;
        }

        int c = kpts[idx].x;
        int r = kpts[idx].y;
        int l = kpts[idx].z;

        pitch = pitch / sizeof(float);
        constexpr float img_scale = 1.f / 255.f;
        constexpr float deriv_scale = img_scale * 0.5f;
        constexpr float second_deriv_scale = img_scale ;
        constexpr float cross_deriv_scale = img_scale * 0.25f;
        constexpr int max_inter_steps{5};

        int optim_idx = 0;

        float local_A[9];
        float local_b[3];
        float local_x[3];
        local_x[0] = 0;
        local_x[1] = 0;
        local_x[2] = 0;

        float dD[3];

        for (; optim_idx < max_inter_steps; optim_idx++) {
            const float* prev = imageBlock + (size_t)(pitch * size.y * (l - 1));
            const float* curr = imageBlock + (size_t)(pitch * size.y * (l));
            const float* next = imageBlock + (size_t)(pitch * size.y * (l + 1));

            local_b[0] = deriv_scale * (curr[r * pitch + c + 1] - curr[r * pitch + c - 1]);
            local_b[1] = deriv_scale * (curr[(r + 1) * pitch + c] - curr[(r - 1) * pitch + c]);
            local_b[2] = deriv_scale * (next[r * pitch + c] - prev[r * pitch + c]);

            float v2 = curr[r * pitch + c] * 2;
            // dxx
            local_A[0] = (curr[r * pitch + c + 1] + curr[r * pitch + c - 1] - v2) * second_deriv_scale;
            local_A[4] = (curr[(r + 1) * pitch + c] + curr[(r - 1) * pitch + c] - v2) * second_deriv_scale;
            local_A[8] = (next[r * pitch + c] + prev[r * pitch + c] - v2) * second_deriv_scale;

            float dxy = (curr[(r+1) * pitch + c + 1] - curr[(r + 1) * pitch + c - 1] -
                         curr[(r-1) * pitch + c + 1] + curr[(r - 1) * pitch + c - 1])*cross_deriv_scale;
            local_A[1] = dxy;
            local_A[3] = dxy;

            float dxs = (next[r * pitch + c + 1] - next[r * pitch + c - 1] -
                         prev[r * pitch + c + 1] + prev[r * pitch + c - 1])*cross_deriv_scale;
            local_A[2] = dxs;
            local_A[6] = dxs;

            float dys = (next[(r+1) * pitch + c] - next[(r - 1) * pitch + c] -
                         prev[(r+1) * pitch + c] + prev[(r - 1) * pitch + c])*cross_deriv_scale;
            local_A[5] = dys;
            local_A[7] = dys;

            solve_3x3_system(local_A, local_b, local_x);

            if (abs(local_x[0]) < 0.5f && abs(local_x[1]) < 0.5f && abs(local_x[2]) < 0.5f) {
                // Converged
                break;
            }

            if (abs(local_x[0]) > size.x || abs(local_x[1]) > size.y || abs(local_x[2]) > 100) {
                // diverged
                mask[idx] = 0;
                return;
            }

            l -= roundf(local_x[2]);
            r -= roundf(local_x[1]);
            c -= roundf(local_x[0]);

            if (l < 1 || l > size.z - 2) {
                mask[idx] = 0;
                return;
            }

            if ( r < sift_cuda::SIFT_IMG_BORDER || r >= size.y - sift_cuda::SIFT_IMG_BORDER ||
                 c < sift_cuda::SIFT_IMG_BORDER || c >= size.x - sift_cuda::SIFT_IMG_BORDER) {
                mask[idx] = 0;
                return;
            }
        }

        if (optim_idx >= sift_cuda::SIFT_MAX_INTER_STEPS) {
            mask[idx] = 0;
            return;
        }

        // Check contrast
        const float* res_prev = imageBlock + (size_t)(pitch * size.y * (l - 1));
        const float* res_curr = imageBlock + (size_t)(pitch * size.y * l);
        const float* res_next = imageBlock + (size_t)(pitch * size.y * (l + 1));
        dD[0] = deriv_scale * (res_curr[r * pitch + c + 1] - res_curr[r * pitch + c - 1]);
        dD[1] = deriv_scale * (res_curr[(r + 1) * pitch + c] - res_curr[(r - 1) * pitch + c]);
        dD[2] = deriv_scale * (res_next[r * pitch + c] - res_prev[r * pitch + c]);
        
        float t = dD[0] * local_x[0] + dD[1] * local_x[1] + dD[2] * local_x[2];
        // local_x need to be inverted
        float contrast = res_curr[r * pitch + c] * img_scale - t * 0.5f;
        if (abs(contrast) * numOctaveLayers < contrastThreshold) {
            mask[idx] = 0;
            return;
        }

        // Check Curvature
        float v2 =   res_curr[r * pitch + c] * 2.f;
        float dxx = (res_curr[r * pitch + c + 1] + res_curr[r * pitch + c - 1] - v2) * second_deriv_scale;
        float dyy = (res_curr[(r + 1) * pitch + c] + res_curr[(r - 1) * pitch + c] - v2) * second_deriv_scale;
        float dxy = (res_curr[(r + 1) * pitch + c + 1] - res_curr[(r + 1) * pitch + c - 1] -
                     res_curr[(r - 1) * pitch + c + 1] + res_curr[(r - 1) * pitch + c - 1]) * cross_deriv_scale;
        float tr  = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;
        if (det <= 0 || tr * tr * edgeThreshold >= (edgeThreshold + 1) * (edgeThreshold + 1) * det) {
            mask[idx] = 0;
            return;
        }

        float mod_val = upscaled ? 2 : 1;
        kpts[idx].x = c * ( 1 << octave ) / mod_val;
        kpts[idx].y = r * ( 1 << octave ) / mod_val;
        kpts[idx].z = l;
        features[idx].x = octave + (l << 8) + (int(roundf(local_x[2] + 0.5) * 255) << 16); // octave
        features[idx].y = sigma * powf(2.f, (l - local_x[2]) / numOctaveLayers) * ( 1 << octave ) * 2; // size
        features[idx].z = abs(contrast); // response
        mask[idx] = 1;
    }

    __global__ void collectKpts(
        const int* mask,
        const int* prefix_sum,
        const float3* kpts,
        float3* output,
        const float4* features,
        float4* output_feature,
        size_t total_size,
        int* collect_size_ptr
    ) {
        int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (x_idx >= total_size) {
            return;
        }

        if (mask[x_idx] == 1) {
            int new_idx = prefix_sum[x_idx];

            if (new_idx >= collect_size_ptr[0]) {
                return;
            }

            output[new_idx - 1] = kpts[x_idx];
            output_feature[new_idx - 1] = features[x_idx];
        }
    }

    __global__ void calOriHistMultiThread(
        const float* imageBlock,
        float3* kpts,
        float4* features,
        int* mask,
        const float2 size,
        const int pitch,
        int* total_num_kpts,
        const int max_expanded_kpts,
        const int oct_idx,
        const float radius_factor,
        const float peak_ratio,
        const float scale_factor,
        bool upscaled
    ) {
        constexpr int warp_size = 32;

        int warpId = threadIdx.x / warp_size;
        int laneId = threadIdx.x % warp_size;
        int kptIdx = blockDim.x * blockIdx.x / warp_size + warpId;

        if (kptIdx < total_num_kpts[0]) {

            int mod = upscaled ? 0 : 1;
            constexpr int numBins = 36;
            constexpr int extraInfo = 4;
            constexpr int usedMemSize = numBins * 2 + extraInfo;
            // 8 pts per block. Since thread is 256, each pts is handled by 32 threads.
            __shared__ float sharedHist[usedMemSize * 8];
            // Each of size 36 * 2 + 4.
            // 0 is radius, 1-3 = x,y,z (base px, py)
            float* shared_info = &(sharedHist[warpId * usedMemSize]);
            float* rawHist = shared_info + extraInfo;
            float* smoothHist = rawHist + numBins;
            float scale = scale_factor * features[kptIdx].y / float(1 << (oct_idx + mod));

            // Initialize
            for (int idx = laneId; idx < usedMemSize; idx += warp_size) {
                shared_info[idx] = 0.f;
            }
            __syncwarp();

            // Precompute and preload data
            if (laneId == 0) {
                shared_info[0] = roundf(radius_factor * scale);
                shared_info[1] = roundf(kpts[kptIdx].x / (1 << oct_idx));
                shared_info[2] = roundf(kpts[kptIdx].y / (1 << oct_idx));
                shared_info[3] = kpts[kptIdx].z;
            }
            __syncwarp();

            int base_px = static_cast<int>(shared_info[1]);
            int base_py = static_cast<int>(shared_info[2]);
            float pl = shared_info[3]; // kpt layer
            float radius = shared_info[0];
            int side = 2 * radius + 1;
            int total_dim = side * side;

            float weight_factor = -0.5f / (scale * scale);
            int h_dim = pitch / sizeof(float);
            const float* image = imageBlock + size_t((size.y * h_dim) * pl);

            for (int idx = laneId; idx < total_dim; idx += warp_size) {
                int py_offset = idx / side - radius;
                int py = base_py + py_offset;
                if (py < 1 || py >= size.y - 1) {
                    continue;
                }

                int px_offset = idx % side - radius;
                int px = base_px + px_offset;
                if (px < 1 || px >= size.x - 1) {
                    continue;
                }

                int py_offset_idx = py * h_dim;

                float dx = image[py_offset_idx + px + 1] - image[py_offset_idx + px - 1];
                float dy = image[py_offset_idx - h_dim + px] - image[py_offset_idx + h_dim + px];
                
                float gradient_mag = __fsqrt_rd(dx * dx + dy * dy);
                float gradient_ori = atan2f(dy, dx) / M_PIf32 * 180.f;

                if (gradient_ori < 0) gradient_ori += 360.f;

                float weight = __expf(weight_factor * (px_offset * px_offset + py_offset * py_offset));
                int hist_index = __float2int_rd(gradient_ori / 360.f * numBins) % numBins;
                atomicAdd(&rawHist[hist_index], weight * gradient_mag);
            }

            __syncwarp();

            for (int idx = laneId; idx < numBins; idx += warp_size) {
                int l_idx = idx > 0 ? idx - 1 : numBins - 1;
                int r_idx = idx < numBins - 1 ? idx + 1 : 0;
                int r2_idx = r_idx < numBins - 1? r_idx + 1 : 0;
                int l2_idx = l_idx > 0 ? l_idx - 1 : numBins - 1;
                // Bank conflicts, but acceptable
                smoothHist[idx] = (6 * rawHist[idx] + 4 * (rawHist[l_idx] + rawHist[r_idx]) + rawHist[l2_idx] + rawHist[r2_idx]) / 16.f;
            }

            __syncwarp();

            if (laneId == 0) {
                float max = 0.f;
                #pragma unroll
                for (int bin_idx = 0; bin_idx < numBins; bin_idx ++) {
                    max = fmaxf(max, smoothHist[bin_idx]);
                }

                float magnitude_threshold = max * peak_ratio;
                int curr_count = 0;
                #pragma unroll
                for (int bin_idx = 0; bin_idx < numBins; bin_idx++) {
                    int l_idx = bin_idx > 0 ? bin_idx - 1 : numBins - 1;
                    int r_idx = bin_idx < numBins - 1 ? bin_idx + 1 : 0;

                    if (
                        smoothHist[bin_idx] > smoothHist[l_idx] &&
                        smoothHist[bin_idx] > smoothHist[r_idx] &&
                        smoothHist[bin_idx] >= magnitude_threshold
                    ) {
                        // TODO (future): Interpolation on angle
                        float orientation = 360.f - float(bin_idx) / numBins * 360.f;
                        int new_idx = kptIdx * numBins + curr_count + total_num_kpts[0];

                        if (new_idx >= max_expanded_kpts) {
                            return;
                        }

                        kpts[new_idx] = kpts[kptIdx];
                        features[new_idx] = features[kptIdx];
                        features[new_idx].w = orientation;
                        mask[new_idx] = 1;
                        curr_count++;
                    }
                }
            }
        }
    }

    __global__ void memsetMask(
        int* mask,
        size_t size
    ) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        if (idx < size) {
            mask[idx] = 0;
        }
    }

    __device__ void unpackOctave(
        const int kpt,
        int* ret_layer,
        float* ret_scale
    ) {
        int oct = kpt & 255;
        int layer = (kpt >> 8) & 255;
        oct = oct < 128 ? oct : (-128 | oct);

        *ret_layer = layer;

        *ret_scale = oct >= 0 ? 1.f / float(1 << oct) : float(1 << -oct);
    }

    __device__ float fast_atan2f(float y, float x) {
        const float ONEQTR_PI = 0.785398163f;
        const float THRQTR_PI = 2.356194490f;
        float r, angle;
        float abs_y = fabsf(y) + 1e-10f;
        
        if (x < 0.0f) {
            r = (x + abs_y) / (abs_y - x);
            angle = THRQTR_PI;
        } else {
            r = (x - abs_y) / (x + abs_y);
            angle = ONEQTR_PI;
        }
        
        angle += (0.1963f * r * r - 0.9817f) * r;
        return (y < 0.0f) ? -angle : angle;
    }

    __device__ void norm(float* ptr, int size, float* norm) {
        float count = 0;
        for (int idx = 0; idx < size; idx++) {
            float val = ptr[idx];
            count = __fmaf_rn(val, val, count);
        }

        *norm = sqrtf(count);
    }

    __device__ void copyToDescriptor(float* hist, float* descriptor) {

        constexpr int d = 4;
        constexpr int n = 8;

        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                int idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);
                hist[idx] += hist[idx + n];
                hist[idx + 1] += hist[idx + n + 1];
                for(int k = 0; k < n; k++ ) {
                    descriptor[(i * d + j) * n + k] = hist[idx + k];
                }
            }
        }
    }

    __device__ void clip_lower(float* desc, int size, float val) {
        for (int idx = 0; idx < size; idx++) {
            desc[idx] = fmin(desc[idx], val);
        }
    }

    __global__ void genDescriptorMultiThread(
        const float* imageBlock,
        const float3* kpt,
        const float4* feature,
        const int num_pts,
        const float scale_multiplier,
        const float3 block_size,
        const size_t pitch,
        half* descriptor
    ) {
        constexpr int warp_size = 128; // not really warp size, num threads to handle 1 pt
        int warpId = threadIdx.x / warp_size;
        int kptIdx = blockDim.x * blockIdx.x / warp_size + warpId;
        if (kptIdx >= num_pts) {
            return;
        }

        int laneId = threadIdx.x % warp_size;
        constexpr int numPtsPerBlock = 256 / warp_size;

        // calcDescriptorsComputer
        float x = kpt[kptIdx].x;
        float y = kpt[kptIdx].y;
        float z = feature[kptIdx].x;
        float size = feature[kptIdx].y;

        int layer;
        float scale;
        int hdim = pitch / sizeof(float);
        unpackOctave(z, &layer, &scale);

        // calcSIFTDescriptor
        // img, ptf, angle, scl
        const float* imgPtr = imageBlock + size_t(hdim * block_size.y * layer);
        x = __float2int_rn(x * scale);
        y = __float2int_rn(y * scale);
        float angle = 360.f - feature[kptIdx].w;
        if (fabs(angle - 360.f) < 1e-6f) {
            angle = 0.f;
        }
        float scl = size * scale * 0.5f;
        // exec

        constexpr int window_width = 4;
        constexpr int num_bins = 8;
        constexpr float sqrt_2 = 1.4142135623730951f;
        constexpr float deg2rad = M_PI / 180.f;
        float cos_t, sin_t;

        __sincosf(angle * deg2rad, &sin_t, &cos_t);

        float bins_per_degree = num_bins / 360.f;
        float exp_scale = -2.f / (window_width * window_width);
        float hist_width = scale_multiplier * scl;
        int radius = __float2int_rn(hist_width * sqrt_2 * (window_width + 1) * 0.5f);
        radius = fmin(
            radius, 
            sqrtf(
                block_size.x * block_size.x + block_size.y * block_size.y
            )
        );
        cos_t /= hist_width;
        sin_t /= hist_width;

        // constexpr int descriptor_width = 128;
        constexpr int tensor_size = 128 + (window_width + 2) * (window_width + 2) * (num_bins + 2);
        half* descriptor_ptr = descriptor + 128 * kptIdx;

        __shared__ float sharedHistBuffer[numPtsPerBlock * tensor_size];

        float* this_buffer = &(sharedHistBuffer[warpId * tensor_size]);
        for (int idx = laneId; idx < tensor_size; idx+=warp_size) {
            this_buffer[idx] = 0.f;
        }
        __syncthreads();

        float* hist_buffer = this_buffer + 128;

        int side = (2 * radius + 1);
        int total_dim = side * side;

        float offset = window_width / 2 - 0.5f;

        for (int idx = laneId; idx < total_dim; idx += warp_size) {
            int idx_i = idx / side - radius;
            int idx_j = idx % side - radius;
            float c_rot = idx_j * cos_t - idx_i * sin_t;
            float r_rot = idx_j * sin_t + idx_i * cos_t;
            float cbin = c_rot + offset;
            float rbin = r_rot + offset;

            int base_py = y + idx_i;
            int base_px = x + idx_j;

            int y_offset = base_py * hdim;

            // Early terminates
            if (
                rbin > -1 && rbin < window_width && cbin > -1 && cbin < window_width &&
                base_px > 0 && base_px < block_size.x && base_py > 0 && base_py < block_size.y
            ) {
                float dx = -imgPtr[y_offset + base_px - 1] + imgPtr[y_offset + base_px + 1];
                float dx2 = dx * dx;
                float dy = imgPtr[(base_py - 1) * hdim + base_px] - imgPtr[(base_py + 1) * hdim + base_px];
                float dy2 = dy * dy;
                float rbin_int; float cbin_int;
                float rbin_frac = modff(rbin, &rbin_int);
                float cbin_frac = modff(cbin, &cbin_int);
                float weight = __expf((c_rot * c_rot + r_rot * r_rot) * exp_scale);

                float grad_mag = __fsqrt_rn(dx2 + dy2);
                float grad_ori = fast_atan2f(dy, dx) / M_PI * 180.f;
                grad_ori = grad_ori < 0 ? grad_ori + 360.f : grad_ori;
                
                float obin = (grad_ori - angle) * bins_per_degree;
                float mag = grad_mag * weight;
                float obin_int; 
                float obin_frac = modff(obin, &obin_int);

                if (obin_int < 0) {
                    obin_int += num_bins;
                }
                if (obin_int >= num_bins) {
                    obin_int -= num_bins;
                }
                int idx = ((rbin_int + 1) * (window_width + 2) + cbin_int + 1) * (num_bins + 2) + obin_int;

                float v_r1 = mag * rbin_frac, v_r0 = mag * (1 - rbin_frac);
                float v_rc11 = v_r1 * cbin_frac, v_rc10 = v_r1 * (1 - cbin_frac);
                float v_rc01 = v_r0 * cbin_frac, v_rc00 = v_r0 * (1 - cbin_frac);
                float one_min_o = (1 - obin_frac);

                float v_rco111 = v_rc11 * obin_frac, v_rco110 = v_rc11 * one_min_o;
                atomicAdd(&hist_buffer[idx + (window_width + 3) * (num_bins + 2)], v_rco110);
                atomicAdd(&hist_buffer[idx + (window_width + 3) * (num_bins + 2) + 1], v_rco111);

                float v_rco101 = v_rc10 * obin_frac, v_rco100 = v_rc10 * one_min_o;
                atomicAdd(&hist_buffer[idx + (window_width + 2) * (num_bins + 2)], v_rco100);
                atomicAdd(&hist_buffer[idx + (window_width + 2) * (num_bins + 2) + 1], v_rco101);

                float v_rco011 = v_rc01 * obin_frac, v_rco010 = v_rc01 * one_min_o;
                atomicAdd(&hist_buffer[idx + (num_bins + 2)], v_rco010);
                atomicAdd(&hist_buffer[idx + (num_bins + 3)], v_rco011);

                float v_rco001 = v_rc00 * obin_frac, v_rco000 = v_rc00 * one_min_o;
                atomicAdd(&hist_buffer[idx], v_rco000);
                atomicAdd(&hist_buffer[idx + 1], v_rco001);
            }
        }

        __syncthreads();

        if (laneId == 0) {
            copyToDescriptor(hist_buffer, this_buffer);
            float norm_val;
            norm(this_buffer, 128, &norm_val);
            const float thr = norm_val * 0.2;
            clip_lower(this_buffer, 128, thr);
            norm(this_buffer, 128, &norm_val);
            float multiplier = fmax(norm_val, 1e-7f);
            hist_buffer[0] = multiplier;
        }

        __syncthreads();
        float norm = hist_buffer[0];
        for (int idx = laneId; idx < 128; idx += warp_size) {
            float new_val = __saturatef(this_buffer[idx] / norm);
            descriptor_ptr[idx] = __float2half(new_val * 512.f);
        }
    }
}
