#include "sift_cuda/interface/Detector.hh"
#include "sift_cuda/sift_func/Match.cuh"
#include "sift_cuda/types/CudaSiftConfig.hh"
#include "cvUtils/Conversion.hh"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>
#include "CLI/CLI.hpp"

#include <filesystem>

std::vector<cv::Mat> loadImagesFromDirectory(const std::string& dirPath) {
    std::vector<std::string> imagePaths;
    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
        std::string extension = entry.path().extension().string();
        if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
            extension == ".bmp" || extension == ".tiff") {
            imagePaths.push_back(entry.path().string());
        }
    }

    std::vector<cv::Mat> images;
    std::sort(imagePaths.begin(), imagePaths.end());
    for (const auto& path : imagePaths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (!img.empty()) {
            images.push_back(img);
        }
    }

    return images;
}

namespace fs = std::filesystem;

int main (int argc, char** argv) {
    CLI::App app{"Script for running one single detectAndCompute and a matching function"};
    std::string path;
    app.add_option("-d,--dir", path, "Path to image directories");
    CLI11_PARSE(app, argc, argv);

    fs::path img_dir(path);

    if (!fs::exists(img_dir)) {
        std::cout << "Directory does not exists." << std::endl;
        return 1;
    }

    std::vector<cv::Mat> input_cv_img = loadImagesFromDirectory(img_dir);
    std::vector<Imagef> host_img;
    std::for_each(input_cv_img.begin(), input_cv_img.end(), [&host_img](cv::Mat mat) {
        host_img.push_back(OpencvUtils::cvMatToImage<float>(mat));
    });

    CudaSiftConfig config;
    config.upscale = false;
    config.numFeatures = 2000;
    config.col_width = host_img.front().m_image_size.col;
    config.row_width = host_img.front().m_image_size.row;
    sift_cuda::Detector detector(config);
    detector.gpuWarmUpAndAllocate();
    CUDA_CHECK(cudaDeviceSynchronize());

    int prev_size = 0;
    std::vector<cv::KeyPoint> prev_cv_kpts;
    int start_idx = 0;
    for (int idx = start_idx; idx < host_img.size(); idx++) {
        detector.detectAndCompute(host_img[idx]);
        detector.copyToHost(false);
        CUDA_CHECK(cudaDeviceSynchronize());
        int curr_size = detector.total_size;
        auto curr_cv_kpts = OpencvUtils::localKptToCvKpt(
            detector.final_kpts,
            detector.final_features,
            curr_size
        );

        if (idx != start_idx) {
            // As subsequent runs started, previous descriptor is moved to `prev_descriptor` 
            const auto matches_host = sift_cuda::matchBruteForce(
                detector.prev_descriptor,
                prev_size, 
                detector.device_descriptor,
                curr_size
            );

            // Do Ops 
            // Here we show example of displaying matched kpts
            {
                std::vector<cv::DMatch> good_matches = OpencvUtils::cvtMatchToDMatch(matches_host);
                cv::Mat matchImg;
                cv::drawMatches(input_cv_img[idx - 1], prev_cv_kpts, input_cv_img[idx], curr_cv_kpts, good_matches, matchImg);

                cv::imshow("Matches", matchImg);
                cv::waitKey(0);
            }
        }
        prev_size = curr_size;
        prev_cv_kpts = curr_cv_kpts;
    }

    return 0;
}
