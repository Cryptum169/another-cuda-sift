load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "data_generation",
    deps = [
        "//sift_cuda/types:types",
        "//sift_cuda/utils:utils",
        "//sift_cuda/image_func:image_func",
        "@zlib//:zlib",
        "@msgpack-c//:msgpack",
    ],
    srcs = [
        "PerfData.cu",
    ],
    hdrs = [
        "PerfData.cuh",
        "Serialization.hpp",
        "CudaTypeSerialization.cuh",
        "DeserializationUtils.cuh",
    ],
    visibility = ["//visibility:public"],
)
