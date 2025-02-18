#ifndef SIFT_CUDA_HOSTINTERFACE_HH
#define SIFT_CUDA_HOSTINTERFACE_HH

#include "sift_cuda/types/CudaImage.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

namespace sift_cuda {
    bool runFilter(
        const CudaImage& input,
        const CudaImage& kernel,
        const CudaImage& expected
    );

    bool runResize(
        const CudaImage& input,
        const CudaImage& expected
    );

    bool runMinus(
        const ImageBlock::ImageView& left,
        const ImageBlock::ImageView& right,
        const ImageBlock::ImageView& expected
    );

    bool runFindPeaks(
        const ImageBlock& dogBlock,
        const float threshold,
        const int img_border,
        const std::vector<float3>& host_expected,
        const int num_to_adjust
    );

    bool runAdjustPts(
        const ImageBlock& block,
        const std::vector<float3>& input,
        const int num_to_adjust,
        const float edgeThreshold,
        const float contrastThreshould,
        const float numOctaveLayers,
        const float sigma,
        const std::vector<float3>& expected_kpt,
        const std::vector<float4>& expected_feature,
        const int expected_num_pts
    );

    bool runOrientationHist(
        const ImageBlock& dogBlock,
        const std::vector<float3>& input_pts,
        const std::vector<float4>& input_feat,
        const int num_to_orient,
        const float radius_factor,
        const float peak_ratio,
        const float scale_factor,
        const std::vector<float3>& expected_kpt,
        const std::vector<float4>& expected_feature,
        const int expected_num_pts
    );

    bool runDescriptor(
        const ImageBlock& gaussian_block,
        const std::vector<float3>& input_pts,
        const std::vector<float4>& input_feat,
        const int num_to_extract,
        const std::vector<half> expected_descriptor,
        const float scale_multiplier
    );
}

#endif
