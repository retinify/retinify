# Installation
retinify supports Linux (Ubuntu) and can be easily installed using the provided `build.sh` script.  
This script generates a Debian package, enabling clean installation, easy updating, and simple removal using standard package management tools.  

# 1. Dependencies
- [**GCC 11 or later**](https://gcc.gnu.org/releases.html)
- [**CMake 3.14 or later**](https://cmake.org/download/)
- [**OpenCV 4.x**](https://opencv.org/releases/)
  
@note
OpenCV is only required when building `retinify::tools` with the `--tools` option.  
If you do not use the `--tools` option, OpenCV is not required.
  
# 2. Clone the retinify repository.
```bash
git clone --recurse-submodules https://github.com/retinify/retinify.git
cd retinify
```

# 3. Install retinify
Build retinify and install its Debian package.  
Select the hardware backend (e.g., NVIDIA GPU or CPU) to be used for acceleration, then run the installation script with the appropriate options.
  
@note
If you do not add `--install`, a Debian package will just be created in the build directory.  
  
## 3.1 with TensorRT
The following additional libraries are required.
  
| Libraries | [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) | [cuDNN](https://developer.nvidia.com/cudnn-archive) | [TensorRT](https://developer.nvidia.com/tensorrt) |
| :-------- | :-------------------------------------------------------- | :-------------------------------------------------- | :------------------------------------------------ |
| Versions  | `12.x`                                                    | `9.x`                                               | `10.x`                                            |
  
```bash
./build.sh --install --tensorrt --tools
```

## 3.2 with CPU
```bash
./build.sh --install --cpu --tools
```

# 4. Verify Installation
You can check whether the Debian package is installed using the following script.
```bash
dpkg -s libretinify-dev
```
  
# 5. Uninstall retinify
To uninstall the Debian package, use the following script.
```bash
sudo apt remove libretinify-dev
```
