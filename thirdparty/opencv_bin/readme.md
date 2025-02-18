## Install opencv to here

```sh
cd opencv/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=../../opencv_bin ..
make install -j
```