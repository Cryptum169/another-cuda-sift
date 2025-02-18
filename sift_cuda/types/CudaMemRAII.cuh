#pragma once

#include <cuda_runtime.h>

// std:err
#include <iostream>

/*
 * Checks for errors from CUDA
 */
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s: %s\n", \
                __FILE__, __LINE__, cudaGetErrorName(error), \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

namespace cuda_raii {
    struct CudaMemDeleter {
        void operator()(float* ptr) { 
            CUDA_CHECK(
                cudaFree(ptr)
            );
        }
    };

    struct CudaArrayDeleter {
        void operator()(cudaArray_t ptr) { 
            if (ptr != nullptr) {
                CUDA_CHECK(
                    cudaFreeArray(ptr)
                );
            }
         };
    };

    /*
     * For speedy transfer and access between device and host
     */
    template <typename Data_T>
    struct PinnedMemory {
        Data_T* pinned_data;
        bool allocated;

        void allocate(int size) {
            cudaMallocHost(
                &pinned_data,
                size * sizeof(Data_T)
            );
            allocated = true;
        }

        ~PinnedMemory() {
            if (allocated) {
                cudaFreeHost(pinned_data);
            }
        }
    };
}
