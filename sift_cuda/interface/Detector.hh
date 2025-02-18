#pragma once

#include "sift_cuda/types/CudaSiftConfig.hh"
#include "sift_cuda/types/CudaImage.cuh"
#include "sift_cuda/types/KeyPoint.cuh"
#include "sift_cuda/types/SiftData.cuh"

#include "sift_cuda/utils/CudaStreamPool.cuh"
#include "sift_cuda/utils/CudaNavGraphManager.cuh"

#include "sift_cuda/perf/PerfData.cuh"
#include "sift_cuda/perf/Serialization.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <vector>
#include <cmath>

namespace sift_cuda {

class Detector {
    public:
        Detector(const CudaSiftConfig& config) : m_config(config) {
            m_nOctaves = round(std::log(double(std::min(m_config.col_width * 2, m_config.row_width * 2))) / std::log(2.) - 2) + 1;
            std::cout << "nOctaves: " << m_nOctaves << ". " << m_config.col_width << ", " << m_config.row_width << std::endl;
        }

        bool gpuWarmUpAndAllocate();
        void detectAndCompute(const Imagef& image);

        // There was a reason to make them inline
        // I unfortunately have already forgot why
        inline void createInitialImage();
        inline void getGaussianPyramid();
        inline void getDogPyramid();
        inline void getKeyPoints();
        inline void adjustKpts();
        void adjustOrientationAndCleanup();
        void extractDescriptor();
        void collectAcrossOctaves();
        void copyToHost(bool descriptor);

        // This functions enables checkpoint saves of intermediate states
        // Used for individually benchmarking functions
        void setDataGen(const std::string & path) {
            m_perf_datagen = true;
            m_debug_path = path;
        };

        // Results
        thrust::device_vector<float3> device_kpts;
        thrust::device_vector<float4> device_features;
        thrust::device_vector<half> device_descriptor;
        thrust::device_vector<half> prev_descriptor;
        thrust::host_vector<float3> final_kpts{};
        thrust::host_vector<float4> final_features{};
        thrust::host_vector<half> descriptors{};
        int max_kpts = 5000;
        int total_size{0};

    private:
        // spin-up function, preallocate memories on GPU
        void allocateInitialImg();
        void allocateGaussianPyramid();
        void allocateDoGPyramid();
        void allocateExtremas();
        // Data containing allocated memory
        SiftData m_data;

        CudaSiftConfig m_config{};
        bool m_initialized{false};
        int m_nOctaves{0};
        int m_init_col;
        int m_init_row;

        // Cuda controllers
        sift_cuda::CudaStreamPool streamManager;
        sift_cuda::CudaNavGraphManager gaussianGraphManager;
        sift_cuda::CudaNavGraphManager dogGraphManager;
        sift_cuda::CudaNavGraphManager kptGraphManager;
        sift_cuda::CudaNavGraphManager kptProcessGraphManager;
        sift_cuda::CudaNavGraphManager orientGraphManager;

        // Benchmark:
        bool m_perf_datagen{false};
        std::string m_debug_path;
        SerializedInput m_perf_input;
        SerializedExpected m_perf_expected;
        SerializedParams m_perf_params;

        // pinned buffer
        cuda_raii::PinnedMemory<int> pinned_memory;
};

}
