load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "sift_func",
    hdrs = [
        "SiftOps.cuh",
        "Match.cuh",
    ],
    srcs = [
        "SiftOps.cu",
        "Match.cu",
    ],
    deps = [
        "//sift_cuda/types:types",
        "//sift_cuda/image_func:image_func",
    ],
    visibility = ["//visibility:public"]
)
