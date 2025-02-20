load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "interface",
    hdrs = [
        "HostInterface.hh",
        "Detector.hh",
    ],
    srcs = [
        "HostInterface.cu",
        "Detector.cu",
    ],
    deps = [
        "//sift_cuda/perf:data_generation",
        "//sift_cuda/types:types",
        "//sift_cuda/image_func:image_func",
        "//sift_cuda/utils:utils",
        "//sift_cuda/sift_func:sift_func",
    ],
    visibility = ["//visibility:public"],
    copts = ["-std=c++20"],
)
