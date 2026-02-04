@page cpp C++ Docs


## Installation

Please refer to the [Installation](@ref installation) section for details.  


## Learning by example

Example code is available in the following repository:  
👉 [https://github.com/retinify/retinify-opencv-example](https://github.com/retinify/retinify-opencv-example)  


## Key classes

Retinify is written in C++ and defined under the `retinify::` namespace.  
The following classes form the core API:

- `retinify::Status`  
  Represents the result of an operation.

- `retinify::Pipeline`  
  The main interface for performing stereo matching.

Retinify does not use C++ exceptions; all operations report their results via `retinify::Status`.


## Using retinify with CMake

To use retinify in your CMake project, find the package and link against the provided target:  
```cmake
find_package(retinify REQUIRED)
target_link_libraries(${PROJECT_NAME} retinify::retinify)
```
Note that retinify requires **GCC 11 or later**.


## Writing C++ code

The code in this section uses OpenCV's `cv::Mat` for image data.

@note
Retinify does not define its own image class and does not depend on OpenCV.  
Image data is provided via raw pointers and stride information, allowing arbitrary image representations.

To perform stereo matching with rectified stereo images, use the following code:  

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
auto statusInitialize = pipeline.Initialize(static_cast<std::uint32_t>(leftImage.cols), 
                                            static_cast<std::uint32_t>(leftImage.rows));
if (!statusInitialize.IsOK())
{
    return 1;
}

// EXECUTE STEREO MATCHING
auto statusExecute = pipeline.Execute(leftImage.ptr<std::uint8_t>(), 
                                      leftImage.step[0], 
                                      rightImage.ptr<std::uint8_t>(), 
                                      rightImage.step[0]);
if (!statusExecute.IsOK())
{
    return 1;
}

// RETRIEVE DISPARITY
auto statusRetrieve = pipeline.RetrieveDisparity(disparity.ptr<float>(), disparity.step[0]);
if (!statusRetrieve.IsOK())
{
    return 1;
}
```

When initializing the pipeline, you can provide calibration parameters to enable the following features:

- Distortion correction and rectification
- Depth and point cloud generation

To perform stereo matching with non-rectified stereo images, use the following code:

```cpp
#include <retinify/retinify.hpp>
#include <opencv2/opencv.hpp>

// LOAD NON-RECTIFIED STEREO INPUT IMAGES
cv::Mat leftImage = cv::imread("path/to/left.png");
cv::Mat rightImage = cv::imread("path/to/right.png");

// PREPARE OUTPUT BUFFERS
cv::Mat leftRectifiedImage = cv::Mat::zeros(leftImage.size(), leftImage.type());
cv::Mat disparity = cv::Mat::zeros(leftImage.size(), CV_32FC1);
cv::Mat depth = cv::Mat::zeros(leftImage.size(), CV_32FC1);
cv::Mat pointCloud = cv::Mat::zeros(leftImage.size(), CV_32FC3);

// LOAD CALIBRATION PARAMETERS
retinify::CalibrationParameters calibParams;
auto statusLoadCalib = retinify::LoadCalibrationParameters("path/to/calib.json", calibParams);
if (!statusLoadCalib.IsOK())
{
    return 1;
}

// CREATE THE STEREO MATCHING PIPELINE
retinify::Pipeline pipeline;

// INITIALIZE THE PIPELINE WITH CALIBRATION PARAMETERS
auto statusInitialize = pipeline.Initialize(static_cast<std::uint32_t>(leftImage.cols), 
                                            static_cast<std::uint32_t>(leftImage.rows),
                                            retinify::PixelFormat::RGB8, 
                                            retinify::DepthMode::ACCURATE, 
                                            calibParams);
if (!statusInitialize.IsOK())
{
    return 1;
}

// EXECUTE STEREO MATCHING
auto statusExecute = pipeline.Execute(leftImage.ptr<std::uint8_t>(), 
                                      leftImage.step[0], 
                                      rightImage.ptr<std::uint8_t>(), 
                                      rightImage.step[0]);
if (!statusExecute.IsOK())
{
    return 1;
}

// RETRIEVE RECTIFIED LEFT IMAGE
auto statusRetrieveLeftRectified = pipeline.RetrieveRectifiedLeftImage(leftRectifiedImage.ptr<std::uint8_t>(), leftRectifiedImage.step[0]);
if (!statusRetrieveLeftRectified.IsOK())
{
    return 1;
}

// RETRIEVE DISPARITY
auto statusRetrieveDisparity = pipeline.RetrieveDisparity(disparity.ptr<float>(), disparity.step[0]);
if (!statusRetrieveDisparity.IsOK())
{
    return 1;
}

// RETRIEVE DEPTH
auto statusRetrieveDepth = pipeline.RetrieveDepth(depth.ptr<float>(), depth.step[0]);
if (!statusRetrieveDepth.IsOK())
{
    return 1;
}

// RETRIEVE POINT CLOUD
auto statusRetrievePointCloud = pipeline.RetrievePointCloud(pointCloud.ptr<float>(), pointCloud.step[0]);
if (!statusRetrievePointCloud.IsOK())
{
    return 1;
}
```