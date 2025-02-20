#include "sift_cuda/types/CudaImage.cuh"
#include "sift_cuda/perf/PerfData.cuh"
#include "sift_cuda/perf/Serialization.hpp"

#include "sift_cuda/interface/HostInterface.hh"
#include "sift_cuda/perf/DeserializationUtils.cuh"

#include "CLI/CLI.hpp"
#include <cuda_runtime.h>

#include <filesystem>

using namespace sift_cuda;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    CLI::App app{"Script for running simple example of all critical functions. \
        Primary use is to feed it into ncu, since it's much easier to benchmark individual functions \
        from here"};

    std::string bin_path;
    app.add_option("-p,--path", bin_path, "Path to data for performance");
    CLI11_PARSE(app, argc, argv);

    fs::path p(bin_path);
    if (!fs::exists(p) || !fs::is_directory(bin_path)) {
        std::cout << "Path either doesn't exist or is not directory" << std::endl;
        return 0;
    }

    SerializedParams params;
    loadCompressed(params,  p / "params.bin");
    SerializedInput inputs;
    loadCompressed(inputs, p /"input.bin");
    SerializedExpected exp;
    loadCompressed(exp, p / "expected.bin");

    ImageBlock gaussianBlock = deserializeToImageBlock(inputs.gaussian_block_pyramid);
    ImageBlock dogBlock = deserializeToImageBlock(inputs.dog_pyramid);

    bool filter, resize, minus, peaks, adj, orie, descr;

    filter = runFilter(
        deserializeToCudaImage(inputs.original_image),
        deserializeToCudaImage(inputs.initial_image_kernel_1d),
        deserializeToCudaImage(exp.blur_expected)
    );
    resize = runResize(
        deserializeToCudaImage(inputs.original_image),
        deserializeToCudaImage(exp.resize_expected)
    );

    minus = runMinus(
        gaussianBlock.getImagePtrAt(1),
        gaussianBlock.getImagePtrAt(0),
        dogBlock.getImagePtrAt(0)
    );

    peaks = runFindPeaks(
        dogBlock,
        params.threshold,
        params.border,
        exp.findpeaks_expected,
        inputs.num_to_adjust
    );

    adj = runAdjustPts(
        dogBlock, 
        inputs.adjustKpts,
        inputs.num_to_adjust,
        params.edgeThreshold,
        params.contrastThreshould,
        params.numOctaveLayers,
        params.sigma,
        exp.adjust_kpt_expected,
        exp.adjust_feature_expected,
        inputs.num_to_orient
    );

    orie = runOrientationHist(
        dogBlock,
        inputs.orientationKpts,
        inputs.orientationFeatures,
        inputs.num_to_extract,
        params.radius_factor,
        params.peak_ratio,
        params.scale_factor,
        exp.orientation_kpt_expected,
        exp.orientation_feature_expected,
        inputs.num_to_extract
    );

    descr = runDescriptor(
        gaussianBlock,
        inputs.descKpts,
        inputs.descFeature,
        inputs.num_to_extract,
        exp.descriptor,
        params.scale_multiplier
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Filter: " << filter << std::endl;
    std::cout << "Resize: " << resize << std::endl;
    std::cout << "Minus: " << minus << std::endl;
    std::cout << "Peaks: " << peaks << std::endl;
    std::cout << "Adjust: " << adj << std::endl;
    std::cout << "orientationHist: " << orie << std::endl;
    std::cout << "descriptor: " << descr << std::endl;


    return 0;
}
