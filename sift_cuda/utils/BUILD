load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "utils",
    hdrs = [
        "CudaStreamPool.cuh",
        "CudaNavGraphManager.cuh",
        "GaussianUtils.hh",
        "CudaMemcpyUtils.cuh",
    ],
    srcs = [
        "GaussianUtils.cc",
        "CudaMemcpyUtils.cu",
    ],
    deps = [
        "//sift_cuda/types:types",
    ],
    visibility = ["//visibility:public"],
    copts = ["-std=c++20"],
)
