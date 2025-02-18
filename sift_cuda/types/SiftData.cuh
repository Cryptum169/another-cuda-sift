#pragma once

#include "CudaImage.cuh"
#include "KeyPoint.cuh"

#include <vector>

namespace sift_cuda {
    
    /*
     * Memory needed to collect keypoints across different octaves, so that
     *    to extract descriptors on batch.
     */
    struct MemLocData {
        thrust::device_vector<int*> prefix_sum_num_pts_for_target;
        thrust::device_vector<int> contiguous_num_pts;
        
        thrust::device_vector<half*> descriptor_octave_begins;
        thrust::device_vector<float3*> kpt_octave_begins;
        thrust::device_vector<float4*> feature_octave_begins;
    };

    /*
     * Memory needed to create keypoint and descriptor on GPU
     */
    struct SiftData {
        CudaImage original_image{};
        CudaImage initial_image_resized{};
        CudaImage initial_image_kernel{};
        CudaImage initial_image_kernel_1d{};

        std::vector<CudaImage> gaussian_blur_temp{};
        std::vector<CudaImage> gaussian_kernels_1d{};
        std::vector<ImageBlock> dog_pyramid{};
        std::vector<ImageBlock> gaussian_block_pyramid;
        std::vector<sift_cuda::KeypointCollections> extremas{};

        MemLocData mem_data;
    };
}
