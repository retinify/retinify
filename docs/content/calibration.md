@page calibration Calibration


## Calibration Parameters

retinify::CalibrationParameters supports the pinhole camera model.

- Intrinsic Parameters: retinify::PinholeIntrinsics

  - Focal lengths: `fx`, `fy`
  - Principal point: `cx`, `cy`
  - Skew coefficient: `skew`

- Distortion Coefficients: retinify::DistortionParameters

  - Radial distortion coefficients: `k1`, `k2`, `k3`, `k4`, `k5`, `k6`
  - Tangential distortion coefficients: `p1`, `p2`

- Rotation matrix: retinify::Mat3x3d

- Translation vector: retinify::Vec3d


## Loading Calibration Parameters

retinify::CalibrationParameters can be loaded directly from a JSON file as follows:

<details>
<summary><strong>Sample Stereo Calibration JSON</strong></summary>

This JSON file defines stereo camera calibration parameters.

```json
{
  // Reprojection error after calibration (optional)
  "calibration_error": 0.0,

  // Calibration timestamp or duration (optional)
  "calibration_time": 0,

  // Image resolution used for calibration
  "image_width": 1241,
  "image_height": 376,

  // Left camera intrinsics
  "left_intrinsics": {
    "fx": 718.856,
    "fy": 718.856,
    "cx": 607.1928,
    "cy": 185.2157,
    "skew": 0.0
  },

  // Right camera intrinsics
  "right_intrinsics": {
    "fx": 718.856,
    "fy": 718.856,
    "cx": 607.1928,
    "cy": 185.2157,
    "skew": 0.0
  },

  // Left camera distortion coefficients
  "left_distortion": {
    "k1": 0.0,
    "k2": 0.0,
    "k3": 0.0,
    "k4": 0.0,
    "k5": 0.0,
    "k6": 0.0,
    "p1": 0.0,
    "p2": 0.0
  },

  // Right camera distortion coefficients
  "right_distortion": {
    "k1": 0.0,
    "k2": 0.0,
    "k3": 0.0,
    "k4": 0.0,
    "k5": 0.0,
    "k6": 0.0,
    "p1": 0.0,
    "p2": 0.0
  },

  // Rotation matrix
  "rotation": [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  ],

  // Translation vector
  "translation": [-0.5371657, 0.0, 0.0]
}
```

</details>

```cpp
// Load calibration parameters from JSON file
retinify::CalibrationParameters calib_params;
retinify::LoadCalibrationParameters("calib.json", calib_params);

// Initialize pipeline with calibration parameters
retinify::Pipeline pipeline;
pipeline.Initialize(image_width, 
                    image_height, 
                    retinify::PixelFormat::RGB8, 
                    retinify::DepthMode::ACCURATE, 
                    calib_params);
```
