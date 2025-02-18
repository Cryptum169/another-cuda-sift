#pragma once

#include "CudaStreamPool.cuh"

#include "sift_cuda/types/CudaImage.cuh"
#include "sift_cuda/types/KeyPoint.cuh"
#include "sift_cuda/types/HostImage.hh"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <vector>

namespace sift_cuda {
    /*
     * Collect the number of valid keypoints for each Octave into a contiguous memory block
     */
    __global__ void copyValidNum(
        int** end_of_prefix_sum,
        int* contiguous_region,
        const int num_octaves
    );

    /*
     * Collect keypoints and descriptors from all the Octaves.
     */
    __global__ void collectKptsAndDescriptor(
        half** src_descriptor,
        float3** src_kpt,
        float4** src_feature,
        half* des_descriptor,
        float3* des_kpt,
        float4* des_feature,
        int* num_pts_each_octave,
        int num_octaves,
        const int max_num_kpts
    );
}