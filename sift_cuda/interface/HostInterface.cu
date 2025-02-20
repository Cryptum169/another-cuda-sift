#include "HostInterface.hh"

#include "sift_cuda/types/CudaImage.cuh"
#include "sift_cuda/image_func/Resize.cuh"
#include "sift_cuda/image_func/Filter.cuh"
#include "sift_cuda/image_func/MatOps.cuh"
#include "sift_cuda/sift_func/SiftOps.cuh"

namespace {

bool comp(const float3& left, const float3& right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool comp(const float4& left, const float4& right) {
    return left.x == right.x && left.y == right.y && left.z == right.z && left.w == right.w;
}

}

namespace sift_cuda {
    bool runFilter(
        const CudaImage& input,
        const CudaImage& kernel,
        const CudaImage& expected
    ) {
        CudaImage output; output.allocate(input.getDim().x, input.getDim().y);
        CudaImage temp; temp.allocate(input.getDim().x, input.getDim().y);

        applyFilter<CudaImage, CudaImage, CudaImage, CudaImage>(input, output, kernel, temp);

        CUDA_CHECK(cudaDeviceSynchronize());

        Imagef gpu_output, host_expected;
        output.copyTo(gpu_output);
        expected.copyTo(host_expected);
        return (gpu_output == host_expected);
    }

    bool runResize(
        const CudaImage& input,
        const CudaImage& expected
    ) {
        CudaImage output; output.allocate(input.getDim().x * 2, input.getDim().y * 2);

        resize_cuda(
            input,
            output,
            output.getDim()
        );

        CUDA_CHECK(cudaDeviceSynchronize());
        Imagef gpu_output, host_expected; 
        output.copyTo(gpu_output);
        expected.copyTo(host_expected);
        return gpu_output == host_expected;
    }

    bool runMinus(
        const ImageBlock::ImageView& left,
        const ImageBlock::ImageView& right,
        const ImageBlock::ImageView& expected
    ) {
        CudaImage output; output.allocate(left.getDim().x, left.getDim().y);

        minus(left, right, output);

        CUDA_CHECK(cudaDeviceSynchronize());
        Imagef gpu_output, host_expected;
        output.copyTo(gpu_output);
        expected.copyTo(host_expected);
        return gpu_output == host_expected;    
    }

    bool runFindPeaks(
        const ImageBlock& dogBlock,
        const float threshold,
        const int img_border,
        const std::vector<float3>& host_expected,
        const int num_to_adjust
    ) {
        sift_cuda::KeypointCollections extrema;
        auto dim = dogBlock.getDim();
        extrema.allocate(dim.x, dim.y, dim.z, num_to_adjust);

        findPeaks3D(
            dogBlock,
            threshold,
            img_border,
            extrema
        );

        int num_pts = extrema.prefix_sum.back();
        thrust::host_vector<float3> host_pt = extrema.candidateKpts;

        bool nums = num_to_adjust == num_pts;

        if (!nums) {
            std::cout << "Host expected: " << num_to_adjust << ", device result: " << num_pts << std::endl;
            return false;
        }

        for (int idx = 0; idx < num_pts - 1; idx++) {
            if (host_expected[idx].x != host_pt[idx].x || 
                host_expected[idx].y != host_pt[idx].y || 
                host_expected[idx].z != host_pt[idx].z) {
                return false;
            }
        }

        return true;
    }

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
    ) {
        int block_size = 256;
        auto dim = block.getDim();

        sift_cuda::KeypointCollections kpt;
        kpt.allocate(dim.x, dim.y, dim.z, input.size() * 5);
        thrust::copy(input.begin(), input.end(), kpt.candidateKpts.begin());

        int execution_size = kpt.collectedKpts.size(); 
        int grid_size = (execution_size + block_size - 1) / block_size;

        int* num_to_adjust_ptr;
        cudaMalloc(&num_to_adjust_ptr, sizeof(int));
        cudaMemcpy(num_to_adjust_ptr, &num_to_adjust, sizeof(int), cudaMemcpyHostToDevice);
        {
            adjustExtrema<<<grid_size, block_size, 0, nullptr>>>(
                thrust::raw_pointer_cast(kpt.candidateKpts.data()),
                num_to_adjust_ptr,
                kpt.candidateKpts.size(),
                0,
                false,
                block.getData(),
                thrust::raw_pointer_cast(kpt.candidateFeatures.data()),
                thrust::raw_pointer_cast(kpt.collectedMask.data()),
                block.getPitch(),
                block.getDim(),
                edgeThreshold,
                contrastThreshould,
                numOctaveLayers,
                sigma
            );

            if (kpt.temp_storage_bytes == 0) {
                cub::DeviceScan::InclusiveSum(
                    nullptr, kpt.temp_storage_bytes,
                    thrust::raw_pointer_cast(kpt.collectedMask.data()),
                    thrust::raw_pointer_cast(kpt.prefix_sum.data()),
                    kpt.collectedMask.size(), nullptr
                );
                kpt.temp_storage.resize(kpt.temp_storage_bytes);
            }

            cub::DeviceScan::InclusiveSum(
                thrust::raw_pointer_cast(kpt.temp_storage.data()),
                kpt.temp_storage_bytes,
                thrust::raw_pointer_cast(kpt.collectedMask.data()),
                thrust::raw_pointer_cast(kpt.prefix_sum.data()),
                kpt.collectedMask.size(), nullptr
            );

            sift_cuda::collectKpts<<<grid_size, block_size, 0, nullptr>>>(
                thrust::raw_pointer_cast(kpt.collectedMask.data()),
                thrust::raw_pointer_cast(kpt.prefix_sum.data()),
                thrust::raw_pointer_cast(kpt.candidateKpts.data()),
                thrust::raw_pointer_cast(kpt.collectedKpts.data()),
                thrust::raw_pointer_cast(kpt.candidateFeatures.data()),
                thrust::raw_pointer_cast(kpt.collectedFeatures.data()),
                kpt.collectedKpts.size(),
                thrust::raw_pointer_cast(kpt.prefix_sum.data() + kpt.collectedMask.size() - 1)
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::host_vector<float3> host_adj_kpt = kpt.collectedKpts;
        thrust::host_vector<float4> host_adj_feat = kpt.collectedFeatures;

        int device_numpts;
        cudaMemcpy(
            &device_numpts,
            thrust::raw_pointer_cast(kpt.prefix_sum.data() + kpt.collectedMask.size() - 1),
            sizeof(int),
            cudaMemcpyDeviceToHost
        );

        if (device_numpts != expected_num_pts) {
            std::cout << "runAdjustPts: returned size neq: " << device_numpts << " vs " << expected_num_pts << std::endl;
            return false;
        }

        for (int idx = 0; idx < device_numpts - 1; idx++) {
            if (!comp(expected_kpt[idx], host_adj_kpt[idx])) {
                std::cout << "kpt: " << idx << std::endl;
                return false;
            }
            if (!comp(expected_feature[idx], host_adj_feat[idx])) {
                std::cout << "feat: " << idx << std::endl;
                return false;
            }
        }

        return true;
    }

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
    ) {
        auto blockDim = dogBlock.getDim();
        sift_cuda::KeypointCollections kpt;
        kpt.allocate(blockDim.x, blockDim.y, blockDim.z, input_pts.size() * 36);
        thrust::copy(input_pts.begin(), input_pts.end(), kpt.collectedKpts.begin());
        thrust::copy(input_feat.begin(), input_feat.end(), kpt.collectedFeatures.begin());
        
        int* device_num_to_orient_ptr;
        cudaMalloc(&device_num_to_orient_ptr, sizeof(int));
        cudaMemcpy(device_num_to_orient_ptr, &num_to_orient, sizeof(int), cudaMemcpyHostToDevice);

        int block_size = 256;
        int memset_grid_size = (kpt.collectedMask.size() + block_size - 1) / block_size;  // ceiling division
        memsetMask<<<memset_grid_size, block_size, 0, nullptr>>>(
            thrust::raw_pointer_cast(kpt.collectedMask.data()),
            kpt.collectedMask.size()
        );

        int collect_size = kpt.collectedKpts.size();
        int hist_launch_size = collect_size * 32;
        int hist_grid = (hist_launch_size + block_size - 1) / block_size;
        sift_cuda::calOriHistMultiThread<<<hist_grid, block_size, 0, nullptr>>>(
            dogBlock.getData(),
            thrust::raw_pointer_cast(kpt.collectedKpts.data()),
            thrust::raw_pointer_cast(kpt.collectedFeatures.data()),
            thrust::raw_pointer_cast(kpt.collectedMask.data()),
            make_float2(blockDim.x, blockDim.y),
            dogBlock.getPitch(),
            device_num_to_orient_ptr,
            collect_size,
            0, 
            radius_factor,
            peak_ratio,
            scale_factor,
            false
        );

        if (kpt.temp_storage_bytes == 0) {
            cub::DeviceScan::InclusiveSum(
                nullptr, kpt.temp_storage_bytes,
                thrust::raw_pointer_cast(kpt.collectedMask.data()),
                thrust::raw_pointer_cast(kpt.prefix_sum.data()),
                kpt.collectedMask.size(), nullptr
            );
            kpt.temp_storage.resize(kpt.temp_storage_bytes);
        }

        cub::DeviceScan::InclusiveSum(
            thrust::raw_pointer_cast(kpt.temp_storage.data()),
            kpt.temp_storage_bytes,
            thrust::raw_pointer_cast(kpt.collectedMask.data()),
            thrust::raw_pointer_cast(kpt.prefix_sum.data()),
            kpt.collectedMask.size(), nullptr
        );

        int grid_size = (collect_size + block_size - 1) / block_size;
        sift_cuda::collectKpts<<<grid_size, block_size, 0, nullptr>>>(
            thrust::raw_pointer_cast(kpt.collectedMask.data()),
            thrust::raw_pointer_cast(kpt.prefix_sum.data()),
            thrust::raw_pointer_cast(kpt.collectedKpts.data()), 
            thrust::raw_pointer_cast(kpt.candidateKpts.data()),
            thrust::raw_pointer_cast(kpt.collectedFeatures.data()), 
            thrust::raw_pointer_cast(kpt.candidateFeatures.data()),
            collect_size,
            thrust::raw_pointer_cast(kpt.prefix_sum.data() + kpt.collectedMask.size() - 1)
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::host_vector<float3> host_adj_kpt = kpt.candidateKpts;
        thrust::host_vector<float4> host_adj_feat = kpt.candidateFeatures;

        int device_numpts;
        cudaMemcpy(
            &device_numpts,
            thrust::raw_pointer_cast(kpt.prefix_sum.data() + kpt.collectedMask.size() - 1),
            sizeof(int),
            cudaMemcpyDeviceToHost
        );

        if (device_numpts != expected_num_pts) {
            std::cout << "runOrientationHist: returned size neq: " << device_numpts << " vs " << expected_num_pts << std::endl;
            return false;
        }

        for (int idx = 0; idx < device_numpts - 1; idx++) {
            if (!comp(expected_kpt[idx], host_adj_kpt[idx])) {
                std::cout << "kpt: " << idx << std::endl;
                std::cout << expected_kpt[idx].x << ", " << expected_kpt[idx].y << ", " << expected_kpt[idx].z << std::endl;
                std::cout << host_adj_kpt[idx].x << ", " << host_adj_kpt[idx].y << ", " << host_adj_kpt[idx].z << std::endl;
                std::cout << "Total count: " << device_numpts << std::endl;
                return false;
            }

            if (!comp(expected_feature[idx], host_adj_feat[idx])) {
                std::cout << "feat: " << idx << std::endl;
                return false;
            }

        }

        return true;
    }

    bool runDescriptor(
        const ImageBlock& gaussian_block,
        const std::vector<float3>& input_pts,
        const std::vector<float4>& input_feat,
        const int num_to_extract,
        const std::vector<half> expected_descriptor,
        const float scale_multiplier
    ) {
        auto blockDim = gaussian_block.getDim();
        sift_cuda::KeypointCollections kpts;
        kpts.allocate(blockDim.x, blockDim.y, blockDim.z, input_pts.size());
        thrust::copy(input_pts.begin(), input_pts.end(), kpts.candidateKpts.begin());
        thrust::copy(input_feat.begin(), input_feat.end(), kpts.candidateFeatures.begin());

        int* device_num_to_extract_ptr;
        cudaMalloc(&device_num_to_extract_ptr, sizeof(int));
        cudaMemcpy(device_num_to_extract_ptr, &num_to_extract, sizeof(int), cudaMemcpyHostToDevice);

        int block_size = 256;
        int extraction_size = num_to_extract * 128;
        int grid_size = (extraction_size + block_size - 1) / block_size;
        genDescriptorMultiThread<<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(gaussian_block.getData()),
            thrust::raw_pointer_cast(kpts.candidateKpts.data()),
            thrust::raw_pointer_cast(kpts.candidateFeatures.data()),
            num_to_extract,
            scale_multiplier,
            gaussian_block.getDim(),
            gaussian_block.getPitch(),
            thrust::raw_pointer_cast(kpts.descriptor.data())
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::host_vector<half> host_descriptor = kpts.descriptor;

        for (int idx = 0; idx < num_to_extract * 128; idx++) {
            if (abs(__half2float(expected_descriptor[idx] - host_descriptor[idx])) > 1) {
                std::cout << "runDescriptor: idx " << idx << std::endl;
                std::cout << "diff " << __half2float(expected_descriptor[idx])
                    << ", " << __half2float(host_descriptor[idx]) << std::endl;
                return false;
            }
        }

        return true;
    }
}
