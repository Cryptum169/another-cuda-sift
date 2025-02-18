#include "Detector.hh"

#include "sift_cuda/perf/Serialization.hpp"
#include "sift_cuda/types/CudaImage.cuh"
#include "sift_cuda/image_func/Filter.cuh"
#include "sift_cuda/image_func/Resize.cuh"
#include "sift_cuda/image_func/MatOps.cuh"
#include "sift_cuda/sift_func/SiftOps.cuh"
#include "sift_cuda/utils/GaussianUtils.hh"
#include "sift_cuda/utils/CudaMemcpyUtils.cuh"

#include "sift_cuda/perf/DeserializationUtils.cuh"

#include <filesystem>

namespace sift_cuda {
    bool Detector::gpuWarmUpAndAllocate() {
        if (m_initialized) {
            return true;
        }

        if (m_config.col_width == 0 || m_config.row_width == 0) {
            std::cerr << "Image width or height not set." << std::endl;
            return false;
        }

        // Warm up
        cudaFree(0);
        allocateInitialImg();
        allocateGaussianPyramid();
        allocateDoGPyramid();
        allocateExtremas();

        streamManager.init((m_config.numOctaveLayers + 2) * 2);

        pinned_memory.allocate(m_data.extremas.size());

        return true;
    }

    void Detector::allocateInitialImg() {
        m_data.original_image.allocate(m_config.col_width, m_config.row_width);

        m_init_col = m_config.col_width;
        m_init_row = m_config.row_width;

        if (m_config.upscale) {
            m_init_col *= 2;
            m_init_row *= 2;
        }

        m_data.initial_image_resized.allocate(m_init_col, m_init_row);

        float sigma_diff = sqrtf(
            fmax(m_config.sigma * m_config.sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f)
        );
        m_data.initial_image_kernel = getGaussianKernel(sigma_diff);

        m_data.initial_image_kernel_1d = get1DGaussian(sigma_diff);
    }

    void Detector::allocateGaussianPyramid() {
        std::vector<float> sigmas(m_config.numOctaveLayers + 3);
        sigmas.front() = m_config.sigma;
        double k = std::pow( 2., 1. / m_config.numOctaveLayers );
        for( int i = 1; i < m_config.numOctaveLayers + 3; i++ )
        {
            double sig_prev = std::pow(k, (double)(i-1)) * m_config.sigma;
            double sig_total = sig_prev*k;
            sigmas[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
        }

        // Could try straight to gaussian - Is slower
        m_data.gaussian_kernels_1d.resize(m_config.numOctaveLayers + 3);
        for (int layer_idx = 0; layer_idx < m_config.numOctaveLayers + 3; layer_idx++) {
            m_data.gaussian_kernels_1d.at(layer_idx) = get1DGaussian(sigmas[layer_idx]);
        }
        m_data.gaussian_blur_temp.resize(m_nOctaves);

        m_data.gaussian_block_pyramid.resize(m_nOctaves);
        for (int octave_idx = 0; octave_idx < m_nOctaves; octave_idx++) {
            auto & block = m_data.gaussian_block_pyramid.at(octave_idx);
            block.allocate(roundf(m_init_col / powf(2, octave_idx)), roundf(m_init_row / powf(2, octave_idx)), m_config.numOctaveLayers + 3);
            auto dim = block.getDim();
            m_data.gaussian_blur_temp.at(octave_idx).allocate(dim.x, dim.y);
        }
    }

    void Detector::allocateDoGPyramid() {
        m_data.dog_pyramid.resize(m_nOctaves);
        for (size_t idx = 0; idx < m_nOctaves; idx++) {
            auto gauss_block_dim = m_data.gaussian_block_pyramid.at(idx).getDim();
            auto& dogBlock = m_data.dog_pyramid.at(idx);
            dogBlock.allocate(gauss_block_dim.x, gauss_block_dim.y, (m_config.numOctaveLayers + 2));
        }
    }

    void Detector::allocateExtremas() {
        m_data.extremas.resize(m_nOctaves);
        thrust::host_vector<int*> num_pts_to_collect(m_nOctaves);
        thrust::host_vector<half*> descriptor_start(m_nOctaves);
        thrust::host_vector<float3*> kpt_start(m_nOctaves);
        thrust::host_vector<float4*> feature_start(m_nOctaves);

        for (int oct_idx = 0; oct_idx < m_nOctaves; oct_idx++) {
            auto & block = m_data.dog_pyramid.at(oct_idx);
            auto & kpt = m_data.extremas.at(oct_idx);
            // Allocate 10 x in case for orientation histogram expansion
            kpt.allocate(block.getPitch() / sizeof(float), block.getDim().y, block.getDim().z - 2, m_config.numFeatures * 10);

            num_pts_to_collect[oct_idx] = thrust::raw_pointer_cast(kpt.prefix_sum.data() + kpt.collectedMask.size() - 1);
            descriptor_start[oct_idx] = thrust::raw_pointer_cast(kpt.descriptor.data());
            kpt_start[oct_idx] = thrust::raw_pointer_cast(kpt.candidateKpts.data());
            feature_start[oct_idx] = thrust::raw_pointer_cast(kpt.candidateFeatures.data());
        }

        m_data.mem_data.prefix_sum_num_pts_for_target = num_pts_to_collect;
        m_data.mem_data.contiguous_num_pts.resize(m_nOctaves);
        m_data.mem_data.descriptor_octave_begins = descriptor_start;
        m_data.mem_data.kpt_octave_begins = kpt_start;
        m_data.mem_data.feature_octave_begins = feature_start;

        device_kpts.resize(m_config.numFeatures);
        device_features.resize(m_config.numFeatures);
        device_descriptor.resize(m_config.numFeatures * 128);
        prev_descriptor.resize(m_config.numFeatures * 128);

        final_kpts.resize(m_config.numFeatures);
        final_features.resize(m_config.numFeatures);
        descriptors.resize(m_config.numFeatures * 128);
    }

    void Detector::detectAndCompute(const Imagef& image) {
        m_data.original_image = image;

        cudaMemcpyAsync(
            thrust::raw_pointer_cast(prev_descriptor.data()),
            thrust::raw_pointer_cast(device_descriptor.data()),
            prev_descriptor.size() * sizeof(half),
            cudaMemcpyDeviceToDevice
        );

        createInitialImage();

        if (m_perf_datagen) {
            m_perf_input.original_image = m_data.original_image;
            m_perf_input.initial_image_kernel_1d = m_data.initial_image_kernel_1d;

            m_data.initial_image_resized.allocate(m_init_col * 2, m_init_row * 2);
            resize_cuda(
                m_data.original_image,
                m_data.initial_image_resized,
                make_float2(
                    m_data.original_image.getDim().x * 2,
                    m_data.original_image.getDim().y * 2
                )
            );

            m_perf_expected.resize_expected = m_data.initial_image_resized;
            m_perf_expected.blur_expected = m_data.gaussian_block_pyramid.front().getImagePtrAt(0);
        }

        getGaussianPyramid();
        getDogPyramid();

        if (m_perf_datagen) {
            m_perf_input.gaussian_block_pyramid = m_data.gaussian_block_pyramid.front();
            m_perf_expected.minus_expected = m_data.dog_pyramid.front();
        }

        getKeyPoints();

        if (m_perf_datagen) {
            m_perf_params.threshold = std::floor(0.5 * m_config.contrastThreshould / m_config.numOctaveLayers * 255 * SIFT_FIXPT_SCALE);
            m_perf_params.border = SIFT_IMG_BORDER;

            m_perf_input.dog_pyramid = m_data.dog_pyramid.front();
            m_perf_input.num_to_adjust = m_data.extremas.front().prefix_sum.back();
            copyToHostVector(m_perf_expected.findpeaks_expected, m_data.extremas.front().candidateKpts);
            copyToHostVector(m_perf_input.adjustKpts, m_data.extremas.front().candidateKpts);
        }

        adjustKpts();

        if (m_perf_datagen) {
            m_perf_params.edgeThreshold = m_config.edgeThreshould;
            m_perf_params.contrastThreshould = m_config.contrastThreshould;
            m_perf_params.numOctaveLayers = m_config.numOctaveLayers;
            m_perf_params.sigma = m_config.sigma;

            copyToHostVector(m_perf_expected.adjust_kpt_expected, m_data.extremas.front().collectedKpts);
            copyToHostVector(m_perf_expected.adjust_feature_expected, m_data.extremas.front().collectedFeatures);

            // m_perf_input.num_to_orient = m_data.extremas.front().prefix_sum.back();
            const auto& kpt = m_data.extremas.front();
            cudaMemcpy(
                &(m_perf_input.num_to_orient),
                thrust::raw_pointer_cast(kpt.prefix_sum.data() + kpt.collectedMask.size() - 1),
                sizeof(int), 
                cudaMemcpyDeviceToHost
            );
            copyToHostVector(m_perf_input.orientationKpts, m_data.extremas.front().collectedKpts);
            copyToHostVector(m_perf_input.orientationFeatures, m_data.extremas.front().collectedFeatures);
            m_perf_params.radius_factor = SIFT_DESCR_SCL_FCTR;
            m_perf_params.peak_ratio = SIFT_ORI_PEAK_RATIO;
            m_perf_params.scale_factor = SIFT_ORI_SIG_FCTR;
        }

        adjustOrientationAndCleanup();

        if (m_perf_datagen) {
            copyToHostVector(m_perf_expected.orientation_kpt_expected, m_data.extremas.front().candidateKpts);
            copyToHostVector(m_perf_expected.orientation_feature_expected, m_data.extremas.front().candidateFeatures);

            m_perf_input.num_to_extract = pinned_memory.pinned_data[0];
            copyToHostVector(m_perf_input.descKpts, m_data.extremas.front().candidateKpts);
            copyToHostVector(m_perf_input.descFeature, m_data.extremas.front().candidateFeatures);
            m_perf_params.scale_multiplier = SIFT_DESCR_SCL_FCTR;
        }
        // TODO: remove duplicates via sort and unique set
        extractDescriptor();

        if (m_perf_datagen) {
            copyToHostVector(m_perf_expected.descriptor, m_data.extremas.front().descriptor);
            std::filesystem::path p(m_debug_path);
            saveCompressed(m_perf_input, p / "input.bin");
            saveCompressed(m_perf_expected, p / "expected.bin");
            saveCompressed(m_perf_params, p / "params.bin");
        }

        collectAcrossOctaves();
        return;
    }

    void Detector::createInitialImage() {

        if (m_config.upscale) {
            resize_cuda(
                m_data.original_image,
                m_data.initial_image_resized,
                make_float2(
                    m_data.original_image.getDim().x * 2,
                    m_data.original_image.getDim().y
                )
            );
            applyFilter<CudaImage, ImageBlock::ImageView, CudaImage, CudaImage>(
                m_data.initial_image_resized,
                m_data.gaussian_block_pyramid.front().getImagePtrAt(0),
                m_data.initial_image_kernel_1d,
                m_data.gaussian_blur_temp.front()
            );
        } else {
            applyFilter<CudaImage, ImageBlock::ImageView, CudaImage, CudaImage>(
                m_data.original_image,
                m_data.gaussian_block_pyramid.front().getImagePtrAt(0),
                m_data.initial_image_kernel_1d,
                m_data.gaussian_blur_temp.front()
            );
        }
    }

    void Detector::getGaussianPyramid() {

        if (gaussianGraphManager.launchable()) {
            gaussianGraphManager.launch();
            gaussianGraphManager.synchronize();
            return;
        }

        cudaStream_t stream = streamManager.getStream();

        gaussianGraphManager.startCapture(stream);

        for (int octave_idx = 0; octave_idx < m_nOctaves; octave_idx++) {
            auto& block = m_data.gaussian_block_pyramid.at(octave_idx);
            for (int layer_idx = 0; layer_idx < m_config.numOctaveLayers + 3; layer_idx++) {
                auto img_view = block.getImagePtrAt(layer_idx);

                if (octave_idx == 0 && layer_idx == 0) {
                    // no op, already loaded
                } else if (layer_idx == 0) {
                    auto last_base_view = m_data.gaussian_block_pyramid.at(octave_idx - 1).getImagePtrAt(m_config.numOctaveLayers);
                    resize_cuda(
                        last_base_view,
                        img_view,
                        make_float2(
                            roundf(last_base_view.getDim().x / 2),
                            roundf(last_base_view.getDim().y / 2)
                        ),
                        stream
                    );
                } else {
                    auto prev_img_view = block.getImagePtrAt(layer_idx - 1); 
                    CudaImage & blur_temp = m_data.gaussian_blur_temp.at(octave_idx);

                    applyFilter<ImageBlock::ImageView, ImageBlock::ImageView, CudaImage, CudaImage>(
                        prev_img_view,
                        img_view,
                        m_data.gaussian_kernels_1d.at(layer_idx),
                        blur_temp,
                        stream
                    );
                }
            }
        }

        gaussianGraphManager.finalize();
        gaussianGraphManager.launch();
        gaussianGraphManager.synchronize();
    }

    void Detector::getDogPyramid() {
        // This way of parallelization runs faster
        if (dogGraphManager.launchable()) {
            dogGraphManager.launch();
            dogGraphManager.synchronize();
            return;
        }

        cudaStream_t stream = streamManager.getStream();
        dogGraphManager.startCapture(stream);

        int block_size = 256;
        for (int oct_idx = 0; oct_idx < m_nOctaves; oct_idx++) {
            auto& block = m_data.dog_pyramid.at(oct_idx);

            auto &kpt = m_data.extremas.at(oct_idx);

            int memset_grid_size = (kpt.collectedMask.size() + block_size - 1) / block_size;  // ceiling division
            memsetMask<<<memset_grid_size, block_size, 0, stream>>>(
                thrust::raw_pointer_cast(kpt.collectedMask.data()),
                kpt.collectedMask.size()
            );

            memsetMask<<<(kpt.mask.size() + block_size - 1) / block_size, block_size, 0, stream>>>(
                thrust::raw_pointer_cast(kpt.mask.data()),
                kpt.mask.size()
            );

            memsetMask<<<(kpt.prefix_sum.size() + block_size - 1) / block_size, block_size, 0, stream>>>(
                thrust::raw_pointer_cast(kpt.prefix_sum.data()),
                kpt.prefix_sum.size()
            );

            auto& gaussian_block = m_data.gaussian_block_pyramid.at(oct_idx);

            for (int layer_idx = 0; layer_idx < m_config.numOctaveLayers + 2; layer_idx++) {

                auto result_image_view = block.getImagePtrAt(layer_idx);
                const auto& img1 = gaussian_block.getImagePtrAt(layer_idx);
                const auto& img2 = gaussian_block.getImagePtrAt(layer_idx + 1);
                sift_cuda::minus(img2, img1, result_image_view, stream);
            }
        }

        dogGraphManager.finalize();
        dogGraphManager.launch();
        dogGraphManager.synchronize();
    }

    void Detector::getKeyPoints() {
        float threshold = std::floor(0.5 * m_config.contrastThreshould / m_config.numOctaveLayers * 255 * SIFT_FIXPT_SCALE);

        cudaStream_t stream = nullptr;
        // First run will allocate the necessary temporary storage for cub InclusiveSum
        // Second run streamCaptures the graph
        // 3rd run is normal execution
        bool mem_allocation{true};
        if (m_data.extremas.front().temp_storage_bytes == 0) {
            stream = streamManager.getStream();
        } else {
            if (kptGraphManager.launchable()) {
                kptGraphManager.launch();
                kptGraphManager.synchronize();
                return;
            }

            mem_allocation = false;
            stream = streamManager.getStream();
            kptGraphManager.startCapture(stream);
        }
        // TODO: Memory issue?
        //    Reversing this caused the inclusiveSum to not work on the first octave. 
        // and it's only the first octave
        for (int oct_idx = m_nOctaves - 1; oct_idx >= 0; oct_idx--) {
            auto & block = m_data.dog_pyramid.at(oct_idx);
            sift_cuda::KeypointCollections& kpt = m_data.extremas.at(oct_idx);

            sift_cuda::findPeaks3D(
                block,
                threshold,
                SIFT_IMG_BORDER,
                kpt,
                stream
            );
        }

        if (mem_allocation) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            kptGraphManager.finalize();
            kptGraphManager.launch();
            kptGraphManager.synchronize();
        }
    }

    void Detector::adjustKpts() {
        // Get num of kpts for each dog block, and also reduces dimension
        constexpr int block_size = 256;

        if (kptProcessGraphManager.launchable()) {
            kptProcessGraphManager.launch();
            kptProcessGraphManager.synchronize();
            return;
        }

        cudaStream_t kptProcessStream = streamManager.getStream();
        kptProcessGraphManager.startCapture(kptProcessStream);

        for (int oct_idx = 0; oct_idx < m_nOctaves; oct_idx++) {
            sift_cuda::KeypointCollections& kpt = m_data.extremas.at(oct_idx);
            auto& block = m_data.dog_pyramid.at(oct_idx);

            int execution_size = kpt.collectedKpts.size(); 
            int grid_size = (execution_size + block_size - 1) / block_size;

            adjustExtrema<<<grid_size, block_size, 0, kptProcessStream>>>(
                thrust::raw_pointer_cast(kpt.candidateKpts.data()),
                thrust::raw_pointer_cast(kpt.prefix_sum.data() + kpt.prefix_sum.size() - 1),
                kpt.candidateKpts.size(),
                oct_idx,
                m_config.upscale,
                block.getData(),
                thrust::raw_pointer_cast(kpt.candidateFeatures.data()),
                thrust::raw_pointer_cast(kpt.collectedMask.data()),
                block.getPitch(),
                block.getDim(),
                m_config.edgeThreshould,
                m_config.contrastThreshould,
                m_config.numOctaveLayers,
                m_config.sigma
            );

            cub::DeviceScan::InclusiveSum(
                thrust::raw_pointer_cast(kpt.temp_storage.data()),
                kpt.temp_storage_bytes,
                thrust::raw_pointer_cast(kpt.collectedMask.data()),
                thrust::raw_pointer_cast(kpt.prefix_sum.data()),
                kpt.collectedMask.size(), kptProcessStream
            );

            sift_cuda::collectKpts<<<grid_size, block_size, 0, kptProcessStream>>>(
                thrust::raw_pointer_cast(kpt.collectedMask.data()),
                thrust::raw_pointer_cast(kpt.prefix_sum.data()),
                thrust::raw_pointer_cast(kpt.candidateKpts.data()),
                thrust::raw_pointer_cast(kpt.collectedKpts.data()),
                thrust::raw_pointer_cast(kpt.candidateFeatures.data()),
                thrust::raw_pointer_cast(kpt.collectedFeatures.data()),
                kpt.collectedKpts.size(),
                thrust::raw_pointer_cast(kpt.prefix_sum.data() + kpt.collectedMask.size() - 1)
            );

            int memset_grid_size = (kpt.collectedMask.size() + block_size - 1) / block_size;
            memsetMask<<<memset_grid_size, block_size, 0, kptProcessStream>>>(
                thrust::raw_pointer_cast(kpt.collectedMask.data()),
                kpt.collectedMask.size()
            );
        }

        kptProcessGraphManager.finalize();
        kptProcessGraphManager.launch();
        kptProcessGraphManager.synchronize();
    }

    void Detector::adjustOrientationAndCleanup() {
        int block_size = 256;

        if (orientGraphManager.launchable()) {
            orientGraphManager.launch();
            orientGraphManager.synchronize();
            return;
        }

        cudaStream_t stream = streamManager.getStream();
        orientGraphManager.startCapture(stream);

        for (int oct_idx = 0; oct_idx < m_nOctaves; oct_idx++) {
            sift_cuda::KeypointCollections& kpt = m_data.extremas.at(oct_idx);
            auto& block = m_data.dog_pyramid.at(oct_idx);

            int collect_size = kpt.collectedKpts.size();
            int hist_launch_size = collect_size * 32;
            int hist_grid = (hist_launch_size + block_size - 1) / block_size;
            sift_cuda::calOriHistMultiThread<<<hist_grid, block_size, 0, stream>>>(
                block.getData(),
                thrust::raw_pointer_cast(kpt.collectedKpts.data()),
                thrust::raw_pointer_cast(kpt.collectedFeatures.data()),
                thrust::raw_pointer_cast(kpt.collectedMask.data()),
                make_float2(block.getDim().x, block.getDim().y),
                block.getPitch(),
                thrust::raw_pointer_cast(kpt.prefix_sum.data() + kpt.collectedMask.size() - 1),
                // This generates new points, so it's necessary to bound it
                collect_size,
                oct_idx, 
                SIFT_DESCR_SCL_FCTR,
                SIFT_ORI_PEAK_RATIO,
                SIFT_ORI_SIG_FCTR,
                m_config.upscale
            );

            int grid_size = (collect_size + block_size - 1) / block_size;
            cub::DeviceScan::InclusiveSum(
                thrust::raw_pointer_cast(kpt.temp_storage.data()),
                kpt.temp_storage_bytes,
                thrust::raw_pointer_cast(kpt.collectedMask.data()),
                thrust::raw_pointer_cast(kpt.prefix_sum.data()),
                kpt.collectedMask.size(), stream
            );

            sift_cuda::collectKpts<<<grid_size, block_size, 0, stream>>>(
                thrust::raw_pointer_cast(kpt.collectedMask.data()),
                thrust::raw_pointer_cast(kpt.prefix_sum.data()),
                thrust::raw_pointer_cast(kpt.collectedKpts.data()), 
                thrust::raw_pointer_cast(kpt.candidateKpts.data()),
                thrust::raw_pointer_cast(kpt.collectedFeatures.data()), 
                thrust::raw_pointer_cast(kpt.candidateFeatures.data()),
                collect_size,
                thrust::raw_pointer_cast(kpt.prefix_sum.data() + kpt.collectedMask.size() - 1)
            );

        }

        int pfx_collect_size = m_nOctaves;
        int collect_grid = (pfx_collect_size + block_size - 1) / block_size;

        copyValidNum<<<pfx_collect_size, collect_grid, 0, stream>>>(
            thrust::raw_pointer_cast(m_data.mem_data.prefix_sum_num_pts_for_target.data()),
            thrust::raw_pointer_cast(m_data.mem_data.contiguous_num_pts.data()),
            m_nOctaves
        );

        cudaMemcpyAsync(
            pinned_memory.pinned_data,
            thrust::raw_pointer_cast(m_data.mem_data.contiguous_num_pts.data()),
            m_nOctaves * sizeof(int),
            cudaMemcpyDeviceToHost,
            stream
        );

        orientGraphManager.finalize();
        orientGraphManager.launch();
        orientGraphManager.synchronize();
    }

    void Detector::extractDescriptor() {
        int block_size = 256;

        for (int oct_idx = 0; oct_idx < m_nOctaves; oct_idx++) {
            int extraction_size = pinned_memory.pinned_data[oct_idx] * 128;
            int grid_size = (extraction_size + block_size - 1) / block_size;
            if (extraction_size == 0) {
                continue;
            }

            auto & kpts = m_data.extremas.at(oct_idx);
            auto & gaussian_block = m_data.gaussian_block_pyramid.at(oct_idx);
            auto stream = streamManager.getStream();

            genDescriptorMultiThread<<<grid_size, block_size, 0, stream>>>(
                thrust::raw_pointer_cast(gaussian_block.getData()),
                thrust::raw_pointer_cast(kpts.candidateKpts.data()),
                thrust::raw_pointer_cast(kpts.candidateFeatures.data()),
                pinned_memory.pinned_data[oct_idx],
                SIFT_DESCR_SCL_FCTR,
                gaussian_block.getDim(),
                gaussian_block.getPitch(),
                thrust::raw_pointer_cast(kpts.descriptor.data())
            );
        }

        streamManager.synchronizeAll();
    }

    void Detector::collectAcrossOctaves() {
        total_size = 0;
        for (int idx = 0; idx < m_nOctaves; idx++) {
            total_size += pinned_memory.pinned_data[idx];
        }

        int block_size = 512;
        int grid_size = (total_size * 128 + block_size - 1) / block_size;
        collectKptsAndDescriptor<<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(m_data.mem_data.descriptor_octave_begins.data()),
            thrust::raw_pointer_cast(m_data.mem_data.kpt_octave_begins.data()),
            thrust::raw_pointer_cast(m_data.mem_data.feature_octave_begins.data()),
            thrust::raw_pointer_cast(device_descriptor.data()),
            thrust::raw_pointer_cast(device_kpts.data()),
            thrust::raw_pointer_cast(device_features.data()),
            thrust::raw_pointer_cast(m_data.mem_data.contiguous_num_pts.data()),
            m_nOctaves,
            m_config.numFeatures
        );
        total_size = std::min(total_size, m_config.numFeatures);
    }

    void Detector::copyToHost(bool descriptor) {
        cudaMemcpyAsync(
            thrust::raw_pointer_cast(final_kpts.data()),
            thrust::raw_pointer_cast(device_kpts.data()),
            total_size * sizeof(float3),
            cudaMemcpyDeviceToHost,
            streamManager.getStream()
        );

        cudaMemcpyAsync(
            thrust::raw_pointer_cast(final_features.data()),
            thrust::raw_pointer_cast(device_features.data()),
            total_size * sizeof(float4),
            cudaMemcpyDeviceToHost,
            streamManager.getStream()
        );

        if (descriptor) {
            cudaMemcpyAsync(
                thrust::raw_pointer_cast(descriptors.data()),
                thrust::raw_pointer_cast(device_descriptor.data()),
                total_size * 128 * sizeof(half),
                cudaMemcpyDeviceToHost,
                streamManager.getStream()
            );
        }

        streamManager.synchronizeAll();
    }
}
