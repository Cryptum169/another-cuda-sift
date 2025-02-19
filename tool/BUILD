load("@rules_cuda//cuda:defs.bzl", "cuda_binary")

cc_binary(
    name = "detection_example",
    deps = [
        "//sift_cuda/interface:interface",
        "//cvUtils:opencv_utils",
        "@opencv//:opencv",
        "@cli11//:cli11",
    ],
    srcs = [
        "detection_example.cc"
    ],
    copts = ["-std=c++20"],
)

cuda_binary(
    name = "extract_and_match_example",
    deps = [
        "//sift_cuda/interface:interface",
        "//sift_cuda/sift_func:sift_func",
        "//cvUtils:opencv_utils",
        "@opencv//:opencv",
        "@cli11//:cli11",
    ],
    srcs = [
        "extract_and_match_example.cc"
    ],
    copts = ["-std=c++20"],
)

cuda_binary(
    name = "perf",
    deps = [
        "//sift_cuda/perf:data_generation",
        "//sift_cuda/types:types",
        "//sift_cuda/image_func:image_func",
        "//sift_cuda/interface:interface",
        "//cvUtils:opencv_utils",
        "@cli11//:cli11",
    ],
    srcs = [
        "perf.cu"
    ],
    visibility = ["//visibility:public"],
    copts = ["-std=c++20"],
)
