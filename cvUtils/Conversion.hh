#ifndef IMAGE_UTILS_CONVERSION_HH
#define IMAGE_UTILS_CONVERSION_HH

#include "sift_cuda/types/HostImage.hh"

#include <thrust/host_vector.h>

#include <cuda_fp16.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>

namespace OpencvUtils {
    /*
     * Conversion functions for CPU Image type from cv::Mat
     * Preserves Image data type.
     */
    template <typename DataType_T>
    Image<DataType_T> cvMatToImage(const cv::Mat& mat);

    /*
     * Conversion functions for CPU Image type to cv::Mat
     * Preserves Image data type.
     */
    template <typename DataType_T>
    cv::Mat imageToCvMat(Image<DataType_T> image);

    /*
     * Custom function to compare two cv::Mats. 
     * 
     * Used in earlier dev to verify behavior of functions. Likely not used currently
     */
    bool areEqual(const cv::Mat& a, const cv::Mat& b);

    /* 
     * Normalize an Image, for some reason turn it into an 8-bit image?
     *
     * Used in earlier dev to verify behavior of functions. Likely not used currently
     */
    template <typename DataType_T>
    Image8U normalize(const Image<DataType_T>& image);

    /*
     * Conversion from thrust host_vector to opencv Keypoint Types
     */
    std::vector<cv::KeyPoint> localKptToCvKpt(
        const thrust::host_vector<float3>& kpts,
        const thrust::host_vector<float4>& features,
        int size = -1
    );

    /*
     * Conversion from descriptor output from CUDA to opencv's way of representing descriptor
     *
     * Descriptors packed in a row-major fashion.
     * i-th row is the descriptor for i-th point, aka index 128-255 for pt index 1.
     */
    template <typename Data_T>
    cv::Mat descriptorToCvMat(
        const thrust::host_vector<Data_T>& descriptors,
        int num_pts
    );

    /*
     * Convert CUDA genearted match to openCV's DMatch
     * 
     * 1. Matching distance not returned.
     * 2. A match index of -1 indicates no match.
     */
    std::vector<cv::DMatch> cvtMatchToDMatch(const std::vector<int>& match);
};

#endif

#include "ConversionImpl.hpp"
