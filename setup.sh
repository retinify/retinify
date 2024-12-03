# GTK4
sudo apt install -y libgtk-4-dev

# libudev
sudo apt update
sudo apt install -y libudev-dev

# V4L2
sudo apt install -y libv4l-dev

# OpenCV
sudo apt install -y libopencv-dev

# CUDA toolkit and cuDNN
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-12-2 cuda-12-2-dev cudnn9-cuda-12

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# TensorRT
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-10.2.0-cuda-12.5_1.0-1_amd64.deb
sudo apt update
sudo apt install -y tensorrt