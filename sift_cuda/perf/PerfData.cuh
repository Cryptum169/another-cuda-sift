#pragma once

#include "CudaTypeSerialization.cuh"

#include "sift_cuda/types/CudaImage.cuh"
#include "sift_cuda/types/KeyPoint.cuh"

#include <msgpack.hpp>

#include <vector>

struct SerializationImage {
    std::vector<float> data;
    int r;
    int c;
    int z;
    int pitch;

    MSGPACK_DEFINE_MAP(data, r, c, z, pitch);

    SerializationImage() = default;
    SerializationImage(const Imagef& image);
    void operator=(const sift_cuda::CudaImage& image);
    void operator=(const sift_cuda::ImageBlock& block);
    void operator=(const sift_cuda::ImageBlock::ImageView& image_slice);
    bool operator==(const SerializationImage& other) const;
};

struct SerializationKeyPoint {
    std::vector<int> collectedMask{};
    std::vector<float3> collectedKpts{};
    std::vector<float4> collectedFeatures{};
        
    std::vector<float3> candidateKpts{};
    std::vector<float4> candidateFeatures{};

    std::vector<float> descriptor;
    std::vector<float> hist_buffer;

    std::vector<int> mask;
    std::vector<int> prefix_sum;
    size_t temp_storage_bytes{0};

    float3 dim;
    MSGPACK_DEFINE_MAP(
        collectedMask,
        collectedKpts,
        collectedFeatures,
        candidateKpts,
        candidateFeatures,
        descriptor,
        hist_buffer,
        mask,
        prefix_sum,
        temp_storage_bytes,
        dim
    );
};

struct SerializedParams {
    // Find peaks
    float threshold;
    int border;

    // adjustKpts
    float edgeThreshold;
    float contrastThreshould;
    float numOctaveLayers;
    float sigma;
    
    // Orientation
    float radius_factor;
    float peak_ratio;
    float scale_factor;

    // extract descriptor
    float scale_multiplier;

    MSGPACK_DEFINE_MAP(
        threshold,
        border,
        edgeThreshold,
        contrastThreshould,
        numOctaveLayers,
        sigma,
        radius_factor,
        peak_ratio,
        scale_factor,
        scale_multiplier
    );
};

struct SerializedInput {
    SerializationImage original_image; // v
    SerializationImage initial_image_kernel_1d; // v

    SerializationImage gaussian_blur_temp{}; // front
    SerializationImage gaussian_kernels_1d{}; // front
    SerializationImage dog_pyramid{}; // front
    SerializationImage gaussian_block_pyramid; // front

    int num_to_adjust;
    std::vector<float3> adjustKpts;
    int num_to_orient;
    std::vector<float3> orientationKpts;
    std::vector<float4> orientationFeatures;
    int num_to_extract;
    std::vector<float3> descKpts;
    std::vector<float4> descFeature;


    MSGPACK_DEFINE_MAP(
        original_image,
        initial_image_kernel_1d,
        gaussian_blur_temp,
        gaussian_kernels_1d,
        dog_pyramid,
        gaussian_block_pyramid,
        num_to_adjust,
        adjustKpts,
        num_to_orient,
        orientationKpts,
        orientationFeatures,
        num_to_extract,
        descKpts,
        descFeature
    );
};

struct SerializedExpected {
    SerializationImage blur_expected; // input: original_image, initial_image_kernel_1d, gaussian_blur_temp
    SerializationImage resize_expected; // input: original_image, dim x 2
    SerializationImage minus_expected; // input: gaussian block idx 0, idx 1;
    std::vector<float3> findpeaks_expected; // input: dogBlock, threshold, img_border

    std::vector<float3> adjust_kpt_expected;
    std::vector<float4> adjust_feature_expected;

    std::vector<float3> orientation_kpt_expected;
    std::vector<float4> orientation_feature_expected;

    std::vector<half> descriptor;

    MSGPACK_DEFINE_MAP(
        blur_expected,
        resize_expected,
        minus_expected,
        findpeaks_expected,
        adjust_kpt_expected,
        adjust_feature_expected,
        orientation_kpt_expected,
        orientation_feature_expected,
        descriptor
    );
};
