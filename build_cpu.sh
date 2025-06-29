mkdir -p build && cd build
cmake -DUSE_NVIDIA_GPU=OFF \
      -DBUILD_TESTS=OFF \
      ..
make
cpack
cd ..