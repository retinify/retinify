# API Reference 

Retinify is written in C++ and defined under the `retinify::` namespace.  
The key classes for using retinify are as follows:
  
- `retinify::Status`  
Holds the result of a function call.
- `retinify::Pipeline`  
Executes stereo matching on rectified image pairs.
  
The `retinify::tools` namespace provides utility functions using OpenCV.  

- `retinify::tools::StereoMatchingPipeline`  
Executes stereo matching on rectified OpenCV image pairs.
  
> Retinify is designed to avoid throwing exceptions. Instead, it uses `retinify::Status` to report and manage the result of each operation.
  