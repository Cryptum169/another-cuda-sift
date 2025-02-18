#pragma once

#include "PerfData.cuh"
#include "sift_cuda/types/CudaImage.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

namespace sift_cuda {

template <typename T>
inline void copyToHostVector(std::vector<T>& host_stl, const thrust::device_vector<T>& device_vector) {
    host_stl.resize(device_vector.size());
    thrust::copy(device_vector.begin(), device_vector.end(), host_stl.begin());
}

CudaImage deserializeToCudaImage(const SerializationImage& input) {
    Imagef data;
    data.m_image_size = {input.r, input.c};
    data.m_data = std::make_shared<std::vector<float>>(input.data);
    
    CudaImage device_img(data);
    return device_img;
}

inline ImageBlock deserializeToImageBlock(const SerializationImage& imageBlock) {
    ImageBlock block;
    block.fromHost(imageBlock.data, imageBlock.r, imageBlock.c, imageBlock.z);
    return block;
}

}
