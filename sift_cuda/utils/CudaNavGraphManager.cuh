#pragma once

namespace sift_cuda {

/*
 * Resource controller for a single CudaGraph
 *   TODO: remove "Nav". Read too many navigation related things recently ...
 */
class CudaNavGraphManager {
    public:
        CudaNavGraphManager() {}

        void startCapture(cudaStream_t stream) {
            m_stream = stream;
            m_graph = nullptr;
            m_executable = nullptr;
            cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal);
        }

        ~CudaNavGraphManager() {
            if (m_executable) {
                cudaGraphExecDestroy(m_executable);
            }
            if (m_graph) {
                cudaGraphDestroy(m_graph);
            }
        }

        // End capture and create executable
        void finalize() {
            cudaStreamEndCapture(m_stream, &m_graph);
            cudaGraphInstantiate(&m_executable, m_graph, nullptr, nullptr, 0);
        }

        // Launch the graph
        void launch() {
            if (m_executable) {
                cudaGraphLaunch(m_executable, m_stream);
            }
        }

        void synchronize() {
            if (m_executable) {
                cudaStreamSynchronize(m_stream);
            }
        }

        bool launchable() {
            if (m_executable) {
                return true;
            } else {
                return false;
            }
        }

        // Prevent copies
        CudaNavGraphManager(const CudaNavGraphManager&) = delete;
        CudaNavGraphManager& operator=(const CudaNavGraphManager&) = delete;

    private:
        cudaStream_t m_stream = nullptr;
        cudaGraph_t m_graph = nullptr;
        cudaGraphExec_t m_executable = nullptr;
};
}
