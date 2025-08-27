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
- 🌐 **Open Source**: Fully customizable and freely available under an open source license.
- 🔥 **High Precision**: Delivers real-time, accurate 3D mapping and object recognition from stereo image input.
- 💰 **Cost Efficiency**: Runs using just cameras, enabling depth perception with minimal hardware cost.
- 🎥 **Camera-Agnostic**: Accepts stereo images from any rectified camera setup, giving you the flexibility to use your own hardware.

## Basic Usage
![pipeline](https://raw.githubusercontent.com/retinify/assets/main/images/pipeline.png)

`retinify::tools` offers OpenCV-compatible utility functions for image and disparity processing.
  
> [!IMPORTANT]
> The core `retinify::Pipeline` is independent of OpenCV and supports various image data types.
  
```cpp
#include <retinify/retinify.hpp>
#include <opencv2/opencv.hpp>

// LOAD INPUT IMAGES
cv::Mat leftImage = cv::imread(<left_image_path>);
cv::Mat rightImage = cv::imread(<right_image_path>);

// PREPARE OUTPUT CONTAINER
cv::Mat disparity;

// CREATE STEREO MATCHING PIPELINE
retinify::tools::StereoMatchingPipeline pipeline;

// INITIALIZE THE PIPELINE
pipeline.Initialize(leftImage.rows, leftImage.cols);

// EXECUTE STEREO MATCHING
pipeline.Run(leftImage, rightImage, disparity);
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
| [![target_tensorrt_badge][]][build_tensorrt_status] | [![build_tensorrt_badge][]][build_tensorrt_status] |
| [![target_jetson_badge][]][build_jetson_status]     | [![build_jetson_badge][]][build_jetson_status]     |
| ![target_hailort_badge]                             | Coming soon                                        |
| ![target_openvino_badge]                            | Coming soon                                        |

<!-- TARGET BADGES -->
[target_tensorrt_badge]: https://img.shields.io/badge/TensorRT-gray?style=flat-square
[target_jetson_badge]: https://img.shields.io/badge/TensorRT(Jetson)-gray?style=flat-square
[target_hailort_badge]: https://img.shields.io/badge/HailoRT-gray?style=flat-square
[target_openvino_badge]: https://img.shields.io/badge/OpenVINO-gray?style=flat-square

<!-- BUILD STATUS BADGES -->
[build_tensorrt_badge]: https://img.shields.io/github/actions/workflow/status/retinify/retinify/build_tensorrt.yml?style=flat-square&label=build
[build_jetson_badge]: https://img.shields.io/github/actions/workflow/status/retinify/retinify/build_jetson.yml?style=flat-square&label=build

<!-- STATUS LINKS -->
[build_tensorrt_status]: https://github.com/retinify/retinify/actions/workflows/build_tensorrt.yml?query=branch%3Amain
[build_jetson_status]: https://github.com/retinify/retinify/actions/workflows/build_jetson.yml?query=branch%3Amain

## Pipeline Latencies
Latency includes the time for image upload, inference, and disparity download, reported as the median over 10000 iterations.  
These measurements were taken using each setting of `retinify::Mode`.  

> [!NOTE]
> Results may vary depending on the execution environment.

| DEVICE \ MODE           | FAST               | BALANCED           | ACCURATE           |
| ----------------------- | ------------------ | ------------------ | ------------------ |
| NVIDIA RTX 3060         | 4.082ms / 245.0FPS | 4.941ms / 202.4FPS | 12.138ms / 82.4FPS |
| NVIDIA Jetson Orin Nano | 18.532ms / 54.0FPS | 25.151ms / 39.8FPS | 49.190ms / 20.3FPS |

## Third-Party
For a list of third-party dependencies, please refer to [NOTICE.md](./NOTICE.md).

## Contact
For commercial inquiries, additional technical support, or any other questions, please feel free to contact us at **[contact@retinify.ai](mailto:contact@retinify.ai)**.  