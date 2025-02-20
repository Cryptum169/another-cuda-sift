#pragma once

#include <vector>

namespace sift_cuda {
    /*
     * Resource controller for keeping track of cudaStreams
     */
    class CudaStreamPool {
    public:
        CudaStreamPool() {}
        
        void init(size_t num_streams) {
            streams.resize(num_streams);
            for (size_t i = 0; i < num_streams; i++) {
                cudaStreamCreate(&streams[i]);
            }
        }

        ~CudaStreamPool() {
            for (auto& stream : streams) {
                cudaStreamDestroy(stream);
            }
        }

        // Get next available stream in a round-robin fashion
        cudaStream_t getStream() {
            cudaStream_t stream = streams[current_idx];
            current_idx = (current_idx + 1) % streams.size();
            return stream;
        }

        // Wait for all streams to complete
        void synchronizeAll() {
            for (auto& stream : streams) {
                cudaStreamSynchronize(stream);
            }
        }

        size_t size() const {
            return streams.size();
        }

        private:
            std::vector<cudaStream_t> streams;
            size_t current_idx = 0;

    };
}