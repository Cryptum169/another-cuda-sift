#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_fp16.h>

namespace sift_cuda {
    // For each dog Block
    class KeypointCollections {
        public:
            // kpts
            thrust::device_vector<int> collectedMask{};

            // We need two sets of here since when we need to write to a new location when
            //   process and reduce one set.
            thrust::device_vector<float3> collectedKpts{};
            thrust::device_vector<float4> collectedFeatures{};
             
            thrust::device_vector<float3> candidateKpts{};
            thrust::device_vector<float4> candidateFeatures{};
            // x = octave, y = size, z = response, w = orientation

            // descriptor
            thrust::device_vector<half> descriptor;

            // mem needed for processing
            thrust::device_vector<int> mask;
            thrust::device_vector<int> prefix_sum;

            // for inclusive sum
            thrust::device_vector<uint8_t> temp_storage;
            size_t temp_storage_bytes{0};

            float3 dim;

            void allocate(
                int x, int y, int z,
                int num_kpt
            );
    };
}
