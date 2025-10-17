@page tutorials Tutorials

This tutorial provides a step-by-step guide to using retinify with OpenCV. Please refer to the following repository for the code examples used throughout the tutorial.  
👉 [Example Repository](https://github.com/retinify/retinify-opencv-template)
  
## 1. Install OpenCV
```
sudo apt install libopencv-dev
```

## 2. Install retinify
Please refer to the [Installation](installation.html) section for details.  

## 3. Quick Demo
This section demonstrates how to perform stereo matching on an arbitrary stereo image pair.  
First, clone the repository used for the demo:  

```bash
git clone https://github.com/retinify/retinify-opencv-example.git
cd retinify-opencv-example
```

Next, build the project:  

```bash
mkdir build
cd build
cmake ..
make
```

An executable will be produced in the build directory. Use it to perform stereo matching. Locate the demo stereo images in the `retinify-opencv-example/img` directory (you may substitute any stereo image pair). Run the executable with the left and right stereo image paths as arguments:

@note
If TensorRT is used as the backend, creation of the engine file may require some time.

```bash
./retinify-opencv-example <left_image_path> <right_image_path>
```

Upon successful execution, the result will be displayed and a file named `disparity.png` will be saved in the same directory as the executable. If you use the images provided in the `retinify-opencv-example/img` directory, a successful run should produce a disparity image similar to the following:

![demo_output](https://raw.githubusercontent.com/retinify/retinify-opencv-example/main/img/disparity.png)
  
## 4. Create a retinify project
We recommend using a CMake-based project when integrating retinify.  
Retinify requires **GCC 11 or later**.  
```cmake
find_package(retinify REQUIRED)
target_link_libraries(${PROJECT_NAME} retinify::retinify)
```

In this tutorial, we will use image data in the form of `cv::Mat`.  
Stereo matching can be performed using the `retinify::Pipeline`.  
  
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
auto statusInitialize = pipeline.Initialize(static_cast<std::uint32_t>(leftImage.cols), static_cast<std::uint32_t>(leftImage.rows));
if (!statusInitialize.IsOK())
{
    return 1;
}

// EXECUTE STEREO MATCHING
auto statusExecute = pipeline.Execute(leftImage.ptr<std::uint8_t>(), leftImage.step[0], rightImage.ptr<std::uint8_t>(), rightImage.step[0]);
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

// SHOW DISPARITY
auto statusColorize = retinify::ColorizeDisparity(disparity.ptr<float>(), disparity.step[0], disparityColored.ptr<std::uint8_t>(), disparityColored.step[0], disparity.cols, disparity.rows, 256.0F);
if (!statusColorize.IsOK())
{
    return 1;
}

cv::cvtColor(disparityColored, disparityColored, cv::COLOR_RGB2BGR);
cv::imshow("disparity", disparityColored);
cv::waitKey(0);
```
