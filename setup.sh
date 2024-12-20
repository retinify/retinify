sudo apt update

# Build tools
sudo apt install -y build-essential cmake

# GTK4
sudo apt install -y libgtk-4-dev

# libudev
sudo apt install -y libudev-dev

# V4L2
sudo apt install -y libv4l-dev

# glm
sudo apt install -y libglm-dev

# OpenCV
sudo apt install -y libopencv-dev

# CUDA toolkit and cuDNN
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda-toolkit-12-6 cudnn9-cuda-12
# sudo apt install nvidia-driver-565

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# TensorRT
os="ubuntu2404"
tag="10.5.0-cuda-12.6"
sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt