#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace sift_cuda {

    std::vector<int> matchBruteForce(
        const thrust::device_vector<half> & des,
        const int num_des, 
        const thrust::device_vector<half> & src,
        const int num_src
    );

    // out_idx initialized to -1, output of -1 means no match
    // score initialized to 0
    __global__ void matchBruteForce(
        const half* const des,
        const int num_des,
        const half* const src,
        const int num_src,
        float* score,
        int* out_idx
    );
}
