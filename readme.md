# Ancu SIFT

An(other)cu(DA) Sift is a CUDA implementation (among many: [PopSift](https://github.com/alicevision/popsift), [cuSIFT](https://github.com/danielsuo/cuSIFT), etc.) of the Scale Invariant Feature Transform (SIFT) algorithm that detects and describes local features in images.

This implementation closely follows the version by opencv, structured largely through the same flow, achieving real-time results without going through extensive optimizations and keeping the code rather readable. 

I started out this repository as a project for me to learn CUDA, I'm also rather bad at template programming. If you see design choices that are poop or are generally considered bad, it's probably because it is. 

## Performance

On an RTX 4070S, matching between 2 sets of 2000 128D descriptor takes just under 1ms. Detect and Compute (not counting memory transfer time):

| 752x480 | 1920x1200 | 1600x900 |
|---|---|---|
| 0.95ms | 3.1ms | 2.5ms |
| 84MiB | 298MiB | 214 MiB|

## Dependencies

Hardware:
This repo is developed on an RTX 4070 Super, the code compiled and succesfully run for `sm_61`. I currently do not have the capacity to test the code on other hardwares.

Software:

0. `bazel` for build. `g++10`, CUDA version 12.6.
1. `rules_cuda` for cuda toolchains with bazel
2. `zlib` and `msgpack-c` for serializing and saving intermediate states of the algorithm.
3. `cli11` for command line argument parsing.

Item 1-3 should be automatically handled by bazel.

4. `OpenCV`, as I used opencv's image utility functions to load the images and draw matches. **To use the examples provided in `tool`, `opencv` is required.**

## Installation
To build OpenCV dependency, do the following:
```sh
# In thirdparty/opencv/build run the following line.
# It should install everything into thirdparty/opencv_bin
$ cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=../../opencv_bin ..
# Install opencv, specify the number of cores to use if you don't want your computer to freeze up 
$ make install -j
```

### Use as third party package

```sh
# In MODULE.bazel
bazel_dep(name = "another-cuda-sift", version = "")
git_override(
    module_name = "another-cuda-sift",
    remote = "https://github.com/Cryptum169/another-cuda-sift.git",
    commit = "commit-sha" #
)

# In BUILD
cc_binary(
    name = "user_program",
    srcs = ["user_program.cc"],
    deps = ["@another-cuda-sift//sift_cuda/interface:interface"],
)
```

## Usage

`tool/extract_and_match_example.cc` provided an example usage. The file here loads a list of images under a directory, detects and computes the descriptor for keypoints on each image, matches keypoints between consecutive images, and plotted the results. 

```cpp
CudaSiftConfig config;
config.col_width = r;
config.row_width = c;
sift_cuda::Detector detector(config);
detector.gpuWarmUpAndAllocate();

for (...) {
    detector.detectAndCompute(host_img[idx]);
    detector.copyToHost( /* descriptor = */ false);
}
```

## Known Issues and FAQ

See [faq.md](doc/faq.md)

## License

SIFT's patent has expired on 2020-03-06. See [Google Patents](https://patents.google.com/patent/US6711293B1/en)
