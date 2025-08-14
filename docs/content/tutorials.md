\page tutorials Tutorials

This tutorial provides a step-by-step guide to using retinify with OpenCV. Please refer to the following repository for the code examples used throughout the tutorial.  
👉 [Example Repository](https://github.com/retinify/retinify-opencv-template)
  
## 1. Install retinify
Please refer to the [Installation](installation.html) section for details.  
  
## 2. Link retinify
We recommend using a CMake-based project when integrating retinify.  
Retinify requires **C++20** and **GCC 11 or later**.  
```cmake
set(CMAKE_CXX_STANDARD 20)

find_package(retinify REQUIRED)

target_link_libraries(${PROJECT_NAME} retinify)
```

## 3. C++ Code
In this tutorial, we will use image data in the form of `cv::Mat`.  
Stereo matching can be performed using the `retinify::tools::StereoMatchingPipeline`.  
  
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
pipeline.Initialize();

// EXECUTE STEREO MATCHING
pipeline.Run(leftImage, rightImage, disparity);

// SHOW DISPARITY
const int maxDisparity = 256;
cv::imshow("disparity", retinify::tools::ColorizeDisparity(disparity, maxDisparity));
cv::waitKey(0);
```
