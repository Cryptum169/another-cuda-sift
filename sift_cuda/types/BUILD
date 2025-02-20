load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "types",
    hdrs = [
        "CudaImage.cuh",
        "CudaSiftConfig.hh",
        "KeyPoint.cuh",
        "CudaMemRAII.cuh",
        "HostImage.hh",
        "SiftData.cuh",
    ],
    srcs = [
        "CudaImage.cu",
        "KeyPoint.cu",
    ],
    visibility = ["//visibility:public"],
    copts = ["-std=c++20"],
)
