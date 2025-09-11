<a href="https://retinify.ai/" style="display:block;">
  <img
    src="https://raw.githubusercontent.com/retinify/assets/main/logo/logo_mobility.gif"
    alt="logo"
    style="width:100%; display:block;"
  />
</a>

# Real-Time AI Stereo Vision Library
  
[![UBUNTU 24.04](https://img.shields.io/badge/-UBUNTU%2024%2E04-orange?style=flat-square&logo=ubuntu&logoColor=white)](https://releases.ubuntu.com/noble/)
[![UBUNTU 22.04](https://img.shields.io/badge/-UBUNTU%2022%2E04-orange?style=flat-square&logo=ubuntu&logoColor=white)](https://releases.ubuntu.com/jammy/)
[![Release](https://img.shields.io/github/v/release/retinify/retinify?sort=semver&style=flat-square&color=blue&label=Release)](https://github.com/retinify/retinify/releases/latest)
[![License](https://img.shields.io/github/license/retinify/retinify?style=flat-square&label=License)](https://github.com/retinify/retinify/blob/main/LICENSE)
![Language](https://img.shields.io/github/languages/top/retinify/retinify?style=flat-square&color=yellow)  
[![X](https://img.shields.io/badge/Follow-@retinify-blueviolet?style=flat-square&logo=x)](https://x.com/retinify)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-@retinify-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/company/retinify)
[![YouTube](https://img.shields.io/badge/Watch-%40retinify-red?style=flat-square&logo=youtube)](https://www.youtube.com/@retinify_ai)
  
Retinify is an advanced AI-powered stereo vision library designed for robotics. It enables real-time, high-precision 3D perception by leveraging GPU and NPU acceleration.  
  
<table style="width:100%;">
  <tr>
    <td style="width:50%;"><img src="https://raw.githubusercontent.com/retinify/assets/main/videos/motion.gif" style="width:100%;" /></td>
    <td style="width:50%;"><img src="https://raw.githubusercontent.com/retinify/assets/main/videos/desk.gif" style="width:100%;" /></td>
  </tr>
</table>

## Why Retinify?
- 🌐 **Open Source**: Fully customizable and freely available under an Apache-2.0 license.
- 🔥 **High Precision**: Delivers real-time, accurate 3D mapping and object recognition from stereo image input.
- ⚡ **Fast Pipeline**: All necessary computations run seamlessly on the GPU, enabling real-time performance.
- 🎥 **Camera-Agnostic**: Accepts stereo images from any rectified camera setup, giving you the flexibility to use your own hardware.
- 💰 **Cost Efficiency**: Runs using just cameras, enabling depth perception with minimal hardware cost.
- 🪶 **Minimal Dependencies**: The pipeline depends only on CUDA Toolkit, cuDNN, and TensorRT, providing a lean and production-grade foundation without unnecessary dependencies.

## Basic Usage
![pipeline](https://raw.githubusercontent.com/retinify/assets/main/images/pipeline.png)
  
> [!IMPORTANT]
> Retinify is independent of OpenCV and supports various image data types.
  
```cpp
#include <retinify/retinify.hpp>
#include <opencv2/opencv.hpp>

// LOAD INPUT IMAGES
cv::Mat leftImage = cv::imread("path/to/left.png");
cv::Mat rightImage = cv::imread("path/to/right.png");

// PREPARE OUTPUT CONTAINER
cv::Mat disparity = cv::Mat::zeros(leftImage.size(), CV_32FC1);

// CREATE STEREO MATCHING PIPELINE
retinify::Pipeline pipeline;

// INITIALIZE THE PIPELINE
pipeline.Initialize(leftImage.cols, leftImage.rows);

// EXECUTE STEREO MATCHING
pipeline.Run(leftImage.ptr<uint8_t>(), leftImage.step[0],   //
             rightImage.ptr<uint8_t>(), rightImage.step[0], //
             disparity.ptr<float>(), disparity.step[0]);
```

## Getting Started
📖 [retinify documentation](https://docs.retinify.ai/) — Developer guide and API reference.

- 🚀 [Installation Guide](https://docs.retinify.ai/installation.html)  
  Step-by-step guide to build and install retinify.

- 🔨 [Tutorials](https://docs.retinify.ai/tutorials.html)  
  Hands-on examples to get you started with real-world use cases.

- 🧩 [API Reference](https://docs.retinify.ai/api.html)  
  Detailed class and function-level documentation for developers.

## Supported Backends
| 🎯 Target                                            | Status                                             |
| --------------------------------------------------- | -------------------------------------------------- |
| [![target-tensorrt-badge][]][build-tensorrt-status] | [![build-tensorrt-badge][]][build-tensorrt-status] |
| [![target-jetson-badge][]][build-jetson-status]     | [![build-jetson-badge][]][build-jetson-status]     |
| ![target-hailort-badge]                             | Coming soon                                        |
| ![target-openvino-badge]                            | Coming soon                                        |

<!-- TARGET BADGES -->
[target-tensorrt-badge]: https://img.shields.io/badge/TensorRT-gray?style=flat-square
[target-jetson-badge]: https://img.shields.io/badge/TensorRT(Jetson)-gray?style=flat-square
[target-hailort-badge]: https://img.shields.io/badge/HailoRT-gray?style=flat-square
[target-openvino-badge]: https://img.shields.io/badge/OpenVINO-gray?style=flat-square

<!-- BUILD STATUS BADGES -->
[build-tensorrt-badge]: https://img.shields.io/github/actions/workflow/status/retinify/retinify/build-tensorrt.yml?style=flat-square&label=build
[build-jetson-badge]: https://img.shields.io/github/actions/workflow/status/retinify/retinify/build-jetson.yml?style=flat-square&label=build

<!-- STATUS LINKS -->
[build-tensorrt-status]: https://github.com/retinify/retinify/actions/workflows/build-tensorrt.yml?query=branch%3Amain
[build-jetson-status]: https://github.com/retinify/retinify/actions/workflows/build-jetson.yml?query=branch%3Amain

## Pipeline Latencies
Latency includes the time for image upload, inference, and disparity download, reported as the median over 10,000 iterations (measured with `retinify::Pipeline`).  
These measurements were taken using each setting of `retinify::Mode`.  

> [!NOTE]
> Results may vary depending on the execution environment.

| DEVICE \ MODE           | FAST               | BALANCED           | ACCURATE           |
| ----------------------- | ------------------ | ------------------ | ------------------ |
| NVIDIA RTX 3060         | 3.925ms / 254.8FPS | 4.691ms / 213.2FPS | 10.790ms / 92.7FPS |
| NVIDIA Jetson Orin Nano | 17.462ms / 57.3FPS | 19.751ms / 50.6FPS | 46.104ms / 21.7FPS |

## Third-Party
For a list of third-party dependencies, please refer to [NOTICE.md](./NOTICE.md).

## Contact
For commercial inquiries, additional technical support, or any other questions, please feel free to contact us at **[contact@retinify.ai](mailto:contact@retinify.ai)**.  