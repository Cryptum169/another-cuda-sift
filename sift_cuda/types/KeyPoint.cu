#include "KeyPoint.cuh"

namespace sift_cuda {
    void KeypointCollections::allocate(
            int x, int y, int z, int kpt_num
        ) {
        int max_num = x * y * z;

        // This number cannot be larger than x * y
        //  1. Because of how I directly ask later kernels to look up maximum
        //     number of collected keypoints
        //  2. Heuristically, not possible to have that many candidate keypoints
        kpt_num = std::min(x * y, kpt_num);

        dim = make_float3(x, y, z);
        candidateKpts.resize(kpt_num);
        candidateFeatures.resize(kpt_num);
        collectedKpts.resize(kpt_num);
        collectedFeatures.resize(kpt_num);

        collectedMask.resize(kpt_num, 0);
        mask.resize(max_num, 0);
        prefix_sum.resize(max_num, 0);

        descriptor.resize(128 * kpt_num);
    }
}
