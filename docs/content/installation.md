@page installation Installation

retinify supports Linux (Ubuntu) and can be easily installed using the provided `build.sh` script.  
This script generates a Debian package, enabling clean installation, easy updating, and simple removal using standard package management tools.  

## Dependencies
Build Toolchain
- [GCC](https://gcc.gnu.org/releases.html): 11 or later
- [CMake](https://cmake.org/download/): 3.14 or later
  
AI Runtime
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive): 12.x, 13.x
- [cuDNN](https://developer.nvidia.com/cudnn-archive): 9.x
- [TensorRT](https://developer.nvidia.com/tensorrt): 10.x

## Clone the retinify repository.
```bash
git clone --recurse-submodules https://github.com/retinify/retinify.git
cd retinify
```

## Install retinify
Build retinify and install its Debian package.
```bash
./build.sh --install --tensorrt
```

## Verify Installation
You can check whether the Debian package is installed using the following script.
```bash
dpkg -s libretinify-dev
```
  
## Uninstall retinify
To uninstall the Debian package, use the following script.
```bash
sudo apt remove libretinify-dev
```
