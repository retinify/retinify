mkdir -p build && cd build
cmake -DUSE_NVIDIA_GPU=ON \
      -DBUILD_TESTS=OFF \
      ..
make
cpack
cd ..