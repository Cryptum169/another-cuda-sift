## Installation

OpenCV
```sh
# In thirdparty/opencv/build
$ cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=../../opencv_bin ..
# Fix all cmake errors
# This line should install everything into thirdparty/opencv_bin
# Specify the number of cores to use if you don't want your computer to freeze up 
$ make install -j
```
