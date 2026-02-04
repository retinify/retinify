<a href="https://retinify.ai/" style="display:block;">
  <img
    src="https://raw.githubusercontent.com/retinify/assets/main/logo/logo_mobility.gif"
    alt="l![C++20](https://img.shields.io/badge/C++20-FFD700?style=flat-square&logo=c%2B%2B&logoColor=black)
ogo"
    style="width:100%; display:block;"
  />
</a>
  
[![UBUNTU 24.04](https://img.shields.io/badge/-UBUNTU%2024%2E04-orange?style=flat-square&logo=ubuntu&logoColor=white)](https://releases.ubuntu.com/noble/)
[![UBUNTU 22.04](https://img.shields.io/badge/-UBUNTU%2022%2E04-orange?style=flat-square&logo=ubuntu&logoColor=white)](https://releases.ubuntu.com/jammy/)
[![JETPACK 6](https://img.shields.io/badge/-JETPACK%206-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://docs.nvidia.com/jetson/jetpack/index.html)
[![Apache 2.0](https://img.shields.io/badge/Apache_2.0-blue?style=flat-square&logo=apache&label=)](https://www.apache.org/licenses/LICENSE-2.0)
![C++](https://img.shields.io/badge/C++-royalblue?style=flat-square&logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/Python-yellow?style=flat-square&logo=python&logoColor=white)  
[![X](https://img.shields.io/badge/Follow-@retinify-blueviolet?style=flat-square&logo=x)](https://x.com/retinify)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-@retinify-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/company/retinify)
[![YouTube](https://img.shields.io/badge/Watch-%40retinify-red?style=flat-square&logo=youtube)](https://www.youtube.com/@retinify_ai)
  
# retinify
>🚀 Real-Time AI Stereo Vision Library

Retinify is an advanced AI-powered stereo vision library designed for robotics. It enables real-time, high-precision 3D perception by leveraging GPU and NPU acceleration.  
  
<table style="width:100%;">
  <tr>
    <td style="width:50%;"><img src="https://raw.githubusercontent.com/retinify/assets/main/videos/motion.gif" style="width:100%;" /></td>
    <td style="width:50%;"><img src="https://raw.githubusercontent.com/retinify/assets/main/videos/desk.gif" style="width:100%;" /></td>
  </tr>
</table>

## Why retinify?
- 🔥 **High Precision**: Delivers real-time, accurate 3D mapping and object recognition from stereo image input.
- ⚡ **Fast Pipeline**: Distortion correction, rectification, stereo matching, and 3D reprojection are fully accelerated on the GPU, enabling real-time performance.
- 🎥 **Camera-Agnostic**: Accepts stereo images from any camera setup, giving you the flexibility to use your own hardware.
- 💰 **Cost Efficiency**: Runs using just cameras, enabling depth perception with minimal hardware cost.
- 🪶 **Minimal Dependencies**: The pipeline depends only on CUDA Toolkit, cuDNN, and TensorRT, providing a lean and production-grade foundation.

## Basic Usage

<details open>
  <summary>Rectified Stereo Images (🐍 Python)</summary>

```python
import retinify
import numpy as np
from PIL import Image

# LOAD RECTIFIED STEREO INPUT IMAGES
left = np.asarray(Image.open("path/to/left.png").convert("RGB"))
right = np.asarray(Image.open("path/to/right.png").convert("RGB"))

# CREATE STEREO MATCHING PIPELINE
pipe = retinify.Pipeline()

# INITIALIZE THE PIPELINE
pipe.initialize(image_width=left.shape[1], 
                image_height=left.shape[0])

# EXECUTE STEREO MATCHING
pipe.execute(left, right)

# RETRIEVE DISPARITY
disparity = pipe.retrieve_disparity()
```

</details>

<details>
  <summary>Non-Rectified Stereo Images (🐍 Python)</summary>

> Using the calibration parameters, the pipeline performs undistortion, rectification, and 3D reprojection.
  
```python
import retinify
import numpy as np
from PIL import Image

# LOAD NON-RECTIFIED STEREO INPUT IMAGES
left = np.asarray(Image.open("path/to/left.png").convert("RGB"))
right = np.asarray(Image.open("path/to/right.png").convert("RGB"))

# LOAD CALIBRATION PARAMETERS
calib_params = retinify.load_calibration_parameters("path/to/calib.json")

# CREATE STEREO MATCHING PIPELINE
pipe = retinify.Pipeline()

# INITIALIZE THE PIPELINE WITH CALIBRATION PARAMETERS
pipe.initialize(image_width=left.shape[1], 
                image_height=left.shape[0],
                pixel_format=retinify.PixelFormat.RGB8,
                depth_mode=retinify.DepthMode.ACCURATE,
                calibration_parameters=calib_params)

# EXECUTE STEREO MATCHING
pipe.execute(left, right)

# RETRIEVE DISPARITY
disparity = pipe.retrieve_disparity()

# RETRIEVE DEPTH
depth = pipe.retrieve_depth()

# RETRIEVE POINT CLOUD
point_cloud = pipe.retrieve_point_cloud()
```

</details>

<details>
  <summary>Rectified Stereo Images (🧬 C++)</summary>

```cpp
#include <retinify/retinify.hpp>
#include <opencv2/opencv.hpp>

// LOAD RECTIFIED STEREO INPUT IMAGES
cv::Mat leftImage = cv::imread("path/to/left.png");
cv::Mat rightImage = cv::imread("path/to/right.png");

// PREPARE OUTPUT BUFFERS
cv::Mat disparity = cv::Mat::zeros(leftImage.size(), CV_32FC1);

// CREATE THE STEREO MATCHING PIPELINE
retinify::Pipeline pipeline;

// INITIALIZE THE PIPELINE
pipeline.Initialize(leftImage.cols, leftImage.rows);

// EXECUTE STEREO MATCHING
pipeline.Execute(leftImage.ptr<uint8_t>(), 
                 leftImage.step[0],
                 rightImage.ptr<uint8_t>(), 
                 rightImage.step[0]);

// RETRIEVE DISPARITY
pipeline.RetrieveDisparity(disparity.ptr<float>(), disparity.step[0]);
```

</details>

<details>
  <summary>Non-Rectified Stereo Images (🧬 C++)</summary>

> Using the calibration parameters, the pipeline performs undistortion, rectification, and 3D reprojection.
  
```cpp
#include <retinify/retinify.hpp>
#include <opencv2/opencv.hpp>

// LOAD NON-RECTIFIED STEREO INPUT IMAGES
cv::Mat leftImage = cv::imread("path/to/left.png");
cv::Mat rightImage = cv::imread("path/to/right.png");

// PREPARE OUTPUT BUFFERS
cv::Mat disparity = cv::Mat::zeros(leftImage.size(), CV_32FC1);
cv::Mat depth = cv::Mat::zeros(leftImage.size(), CV_32FC1);
cv::Mat pointCloud = cv::Mat::zeros(leftImage.size(), CV_32FC3);

// LOAD CALIBRATION PARAMETERS
retinify::CalibrationParameters calibParams;
retinify::LoadCalibrationParameters("path/to/calib.json", calibParams);

// CREATE THE STEREO MATCHING PIPELINE
retinify::Pipeline pipeline;

// INITIALIZE THE PIPELINE WITH CALIBRATION PARAMETERS
pipeline.Initialize(leftImage.cols, 
                    leftImage.rows, 
                    retinify::PixelFormat::RGB8, 
                    retinify::DepthMode::ACCURATE, 
                    calibParams);

// EXECUTE STEREO MATCHING
pipeline.Execute(leftImage.ptr<uint8_t>(), 
                 leftImage.step[0],
                 rightImage.ptr<uint8_t>(), 
                 rightImage.step[0]);

// RETRIEVE DISPARITY
pipeline.RetrieveDisparity(disparity.ptr<float>(), disparity.step[0]);

// RETRIEVE DEPTH
pipeline.RetrieveDepth(depth.ptr<float>(), depth.step[0]);

// RETRIEVE POINT CLOUD
pipeline.RetrievePointCloud(pointCloud.ptr<float>(), pointCloud.step[0]);
```

</details>

## Getting Started
📖 [**retinify documentation**](https://docs.retinify.ai/)

- 📥 [**Installation**](https://docs.retinify.ai/installation.html)  
  Step-by-step installation instructions

- 🎬 [**Demos**](https://docs.retinify.ai/demos.html)  
  Hands-on examples using OpenCV

- 🔨 [**Tutorials**](https://docs.retinify.ai/tutorials.html)  
  Guided explanations of retinify concepts

- 🎯 [**Calibration**](https://docs.retinify.ai/calibration.html)  
  Calibration Parameters Specification

- 🐍 [**Python Docs**](https://docs.retinify.ai/python.html)  
  Python API documentation

- 🧬 [**C++ Docs**](https://docs.retinify.ai/cpp.html)  
  C++ API documentation

- 🤖 [**ROS2 Docs**](https://docs.retinify.ai/ros2.html)  
  ROS2 integration reference

## Supported Backends
| 🎯 Target             | ⚙️ Env           | 📦 Status                                                               |
| -------------------- | --------------- | ---------------------------------------------------------------------- |
| ![target-tensorrt10] | ![env-cuda12]   | [![build-tensorrt10-cuda12-badge]][build-tensorrt10-cuda12-status]     |
| ![target-tensorrt10] | ![env-cuda13]   | [![build-tensorrt10-cuda13-badge]][build-tensorrt10-cuda13-status]     |
| ![target-tensorrt10] | ![env-jetpack6] | [![build-tensorrt10-jetpack6-badge]][build-tensorrt10-jetpack6-status] |

[target-tensorrt10]: https://img.shields.io/badge/-TensorRT%2010-76B900?style=flat-square&logo=nvidia&logoColor=white
[target-hailort]:    https://img.shields.io/badge/-HailoRT-lightgray?style=flat-square
[target-openvino]:   https://img.shields.io/badge/-OpenVINO-lightgray?style=flat-square

[env-cuda12]:   https://img.shields.io/badge/-CUDA%2012.x-76B900?style=flat-square&logo=nvidia&logoColor=white
[env-cuda13]:   https://img.shields.io/badge/-CUDA%2013.x-76B900?style=flat-square&logo=nvidia&logoColor=white
[env-jetpack6]: https://img.shields.io/badge/-JETPACK%206-76B900?style=flat-square&logo=nvidia&logoColor=white
[env-na]:       https://img.shields.io/badge/-TBD-lightgray?style=flat-square

[build-tensorrt10-cuda12-badge]:  https://img.shields.io/github/actions/workflow/status/retinify/retinify/build-tensorrt10-cuda12.yml?style=flat-square&label=build
[build-tensorrt10-cuda13-badge]:  https://img.shields.io/github/actions/workflow/status/retinify/retinify/build-tensorrt10-cuda13.yml?style=flat-square&label=build
[build-tensorrt10-jetpack6-badge]: https://img.shields.io/github/actions/workflow/status/retinify/retinify/build-tensorrt10-jetpack6.yml?style=flat-square&label=build

[build-tensorrt10-cuda12-status]:   https://github.com/retinify/retinify/actions/workflows/build-tensorrt10-cuda12.yml?query=branch%3Amain
[build-tensorrt10-cuda13-status]:   https://github.com/retinify/retinify/actions/workflows/build-tensorrt10-cuda13.yml?query=branch%3Amain
[build-tensorrt10-jetpack6-status]: https://github.com/retinify/retinify/actions/workflows/build-tensorrt10-jetpack6.yml?query=branch%3Amain

[status-planned]: https://img.shields.io/badge/-Planned-lightgray?style=flat-square

## Pipeline Latencies
Latency includes the time for image upload, inference, and disparity download, reported as the median over 10,000 iterations (measured with `retinify::Pipeline`).  
These measurements were taken using each setting of `retinify::DepthMode`.  

> [!NOTE]
> Results may vary depending on the execution environment.

| DEVICE \ MODE           | FAST               | BALANCED           | ACCURATE           |
| ----------------------- | ------------------ | ------------------ | ------------------ |
| NVIDIA RTX 3060         | 3.925ms / 254.8FPS | 4.691ms / 213.2FPS | 10.790ms / 92.7FPS |
| NVIDIA Jetson Orin Nano | 17.462ms / 57.3FPS | 19.751ms / 50.6FPS | 46.104ms / 21.7FPS |

## Third-Party
For a list of third-party dependencies, please refer to [NOTICE.md](./NOTICE.md).

## Contact
For all inquiries, including support, collaboration, please contact:  
[contact@retinify.ai](mailto:contact@retinify.ai)
