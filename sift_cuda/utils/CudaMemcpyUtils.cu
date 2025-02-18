#include "CudaMemcpyUtils.cuh"

namespace sift_cuda {

    __global__ void copyValidNum(
        int** end_of_prefix_sum,
        int* contiguous_region,
        const int num_octaves
    ) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx >= num_octaves) {
            return;
        }

        contiguous_region[idx] = end_of_prefix_sum[idx][0];

        return;
    }

    __global__ void collectKptsAndDescriptor(
        half** src_descriptor,
        float3** src_kpt,
        float4** src_feature,
        half* des_descriptor,
        float3* des_kpt,
        float4* des_feature,
        int* num_pts_each_octave,
        int num_octaves,
        const int max_num_pts
    ) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        int intra_octave_kpt_idx = idx / 128;
        int des_kpt_idx = intra_octave_kpt_idx;
        int descriptor_idx = idx % 128;

        if (des_kpt_idx >= max_num_pts) {
            return;
        }

        int octave_num = 0;
        for (; octave_num < num_octaves; octave_num++) {
            intra_octave_kpt_idx -= num_pts_each_octave[octave_num];
            if (intra_octave_kpt_idx < 0) {
                intra_octave_kpt_idx += num_pts_each_octave[octave_num];
                break;
            }
        }

        if (octave_num == num_octaves) {
            return;
        }

        // With this, `idx` has been reduced from global index 
        // to intra-octave index

        if (descriptor_idx == 0) {
            float3* oct_kpt_base = src_kpt[octave_num];
            float4* oct_feature_base = src_feature[octave_num];

            des_kpt[des_kpt_idx] = oct_kpt_base[intra_octave_kpt_idx];
            des_feature[des_kpt_idx] = oct_feature_base[intra_octave_kpt_idx];
        }

        half* oct_desc_base = src_descriptor[octave_num] + intra_octave_kpt_idx * 128;
        des_descriptor[des_kpt_idx * 128 + descriptor_idx] = oct_desc_base[descriptor_idx];

        return;
    }
}