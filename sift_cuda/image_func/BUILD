load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "image_func",
    hdrs = [
        "Resize.cuh",
        "ResizeImpl.cuh",
        "Filter.cuh",
        "FilterImpl.cuh",
        "MatOps.cuh",
        "MatOpsImpl.cuh",
        "utils.cuh",
    ],
    srcs = [
        "Resize.cu",
        "Filter.cu",
        "MatOps.cu",
        "utils.cu",
    ],
    deps = [
        "//sift_cuda/types:types",
    ],
    copts = ["-std=c++20"],
    visibility = ["//visibility:public"],
)
