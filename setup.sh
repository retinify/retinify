# OpenCV
sudo apt-get update
sudo apt -y install build-essential gcc g++ make libtool texinfo dpkg-dev pkg-config
sudo apt -y install git cmake cmake-curses-gui cmake-gui curl
sudo apt -y install  libceres-dev libceres2
sudo apt -y build-dep libopencv-dev
sudo apt -y install libgtk2.0-dev
sudo apt -y install openalpr openalpr-utils libopenalpr-dev
sudo apt -y install openni2-utils libopenni2-dev
sudo apt -y install libpcl-dev
sudo apt -y install libjasper-dev libleveldb-dev liblmdb-dev
sudo apt -y install libatlas-base-dev libopenblas-dev liblapack-dev libtbb-dev libeigen3-dev

git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$HOME/local/ \
      -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -DBUILD_JAVA=OFF \
      -DBUILD_EXAMPLES=OFF \
      ..
make -j$(nproc)
make install
sudo ldconfig

echo "export CMAKE_PREFIX_PATH=$HOME/local/:$CMAKE_PREFIX_PATH" >> $HOME/.bashrc
source ~/.bashrc

cd ../..

# CUDA toolkit and cuDNN
sudo apt-get install nvidia-driver-550

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-2
sudo apt-get -y install cuda-12-2-dev
sudo apt-get -y install cudnn9-cuda-12

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# TensorRT
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-10.2.0-cuda-12.5_1.0-1_amd64.deb
sudo apt-get update
sudo apt-get install -y tensorrt