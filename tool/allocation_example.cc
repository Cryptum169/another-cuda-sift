#include "sift_cuda/interface/Detector.hh"
#include "sift_cuda/types/CudaSiftConfig.hh"

#include "cvUtils/Conversion.hh"

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>

#include "CLI/CLI.hpp"

#include <filesystem>

namespace fs = std::filesystem;

int main (int argc, char** argv) {
    CLI::App app{"Script for running detectAndCompute on one single image n times, could use nsys and nsys-ui to benchmark with this."};

    std::string img_path, debug_path;
    app.add_option("-p,--path", img_path, "Path to left img");
    app.add_option("-d,--debug_path", debug_path, "Path to debug directory");
    CLI11_PARSE(app, argc, argv);

    fs::path fs_img_path(img_path);
    cv::Mat img = cv::imread(fs_img_path, cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cout << "Image DNE" << std::endl;
        return 1;
    }
    Imagef converted = OpencvUtils::cvMatToImage<float>(img);

    CudaSiftConfig config;
    config.upscale = false;
    config.col_width = converted.m_image_size.col;
    config.row_width = converted.m_image_size.row;
    std::cout << "r: " << config.row_width << ", c: " << config.col_width << std::endl;

    sift_cuda::Detector detector(config);
    detector.gpuWarmUpAndAllocate();
    if (!debug_path.empty()) {
        detector.setDataGen(debug_path);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int idx = 0; idx < 1; idx++) {
        detector.detectAndCompute(converted);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    return 0;
}
