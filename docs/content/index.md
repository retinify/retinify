# Overview
Retinify is an advanced AI-powered stereo vision library designed for robotics.  
It enables real-time, high-precision 3D perception by leveraging GPU and NPU acceleration.

@note
The source code is publicly available:  
👉 [**View on GitHub**](https://github.com/retinify/retinify)
  
[![UBUNTU 24.04](https://img.shields.io/badge/-UBUNTU%2024%2E04-orange?style=flat-square&logo=ubuntu&logoColor=white)](https://releases.ubuntu.com/noble/)
[![UBUNTU 22.04](https://img.shields.io/badge/-UBUNTU%2022%2E04-orange?style=flat-square&logo=ubuntu&logoColor=white)](https://releases.ubuntu.com/jammy/)
[![JETPACK 6](https://img.shields.io/badge/-JETPACK%206-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://docs.nvidia.com/jetson/jetpack/index.html)
[![Apache 2.0](https://img.shields.io/badge/Apache_2.0-blue?style=flat-square&logo=apache&label=)](https://www.apache.org/licenses/LICENSE-2.0)
![C++](https://img.shields.io/badge/C++-royalblue?style=flat-square&logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/Python-yellow?style=flat-square&logo=python&logoColor=white)  
[![X](https://img.shields.io/badge/Follow-@retinify-blueviolet?style=flat-square&logo=x)](https://x.com/retinify)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-@retinify-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/company/retinify)
[![YouTube](https://img.shields.io/badge/Watch-%40retinify-red?style=flat-square&logo=youtube)](https://www.youtube.com/@retinify_ai)


### The retinify pipeline

<table style="width:100%;">
  <tr>
    <td style="width:50%;"><img src="https://raw.githubusercontent.com/retinify/assets/main/images/vo_pipeline.png" style="width:100%;" /></td>
    <td style="width:50%;"><img src="https://raw.githubusercontent.com/retinify/assets/main/videos/gpu_pipeline.gif" style="width:100%;" /></td>
  </tr>
</table>

The main functionality of retinify is accessible through retinify::Pipeline.
  
**GPU-Based Computing:**  
All processing required for stereo depth estimation — including image remapping (undistortion and rectification), stereo matching, and 3D reprojection — is executed entirely on the GPU, enabling efficient and real-time performance.

**AI-Powered Stereo Matching:**  
Stereo matching is computed using deep learning–based models, delivering significantly higher accuracy and robustness compared to traditional algorithms.

**Flexible Integration:**  
The pipeline is camera-agnostic and supports any stereo camera setup. Input images are handled via raw pointers and strides, allowing seamless integration without dependencies on specific image processing libraries.


### Getting started

- [Installation](@ref installation)  
- [Demos](@ref demos)  
- [Tutorials](@ref tutorials)  
- [Calibration](@ref calibration)
- [Python Docs](@ref python)  
- [C++ Docs](@ref cpp)  
- [ROS2 Docs](@ref ros2)  


### Contact

For all inquiries, including support, collaboration, please contact:  
[contact@retinify.ai](mailto:contact@retinify.ai)
