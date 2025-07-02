mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr \
      -DUSE_NVIDIA_GPU=ON \
      -DBUILD_TESTS=OFF \
      ..
make
cpack
cd ..