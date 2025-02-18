load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "opencv_utils",
    srcs = [
        "Conversion.cc"
    ],
    hdrs = [
        "Conversion.hh",
        "ConversionImpl.hpp",
    ],
    deps = [
        "//sift_cuda/types:types",
        "@opencv//:opencv"
    ],
    visibility = ["//visibility:public"],
)
