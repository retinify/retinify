// Copyright (C) 2024 retinify project team. All rights reserved.
//
// This file is part of retinify.
//
// retinify is free software: you can redistribute it and/or modify it under the terms of the
// GNU Affero General Public License as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// retinify is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with retinify.
// If not, see <https://www.gnu.org/licenses/>.

#pragma once
#include <map>
#include <memory>
#include <opencv2/core.hpp>
namespace retinify
{
/// @brief Image data with timestamp
struct ImageData
{
    cv::Mat image_;
    float timestamp_;
};
/// @brief Stereo Image data including left and right images, disparity and point cloud
struct StereoImageData
{
    ImageData left_;
    ImageData right_;
    cv::Mat disparity_;
    cv::Mat pcd_;
};
/// @brief Device data
struct DeviceData
{
    std::string node_;
    std::string name_;
    std::string capabilities_;
    std::string serialNumber_;
    std::map<uint32_t, std::vector<cv::Size>> formats_;
};
class CalibrationData
{
  public:
    CalibrationData();
    ~CalibrationData();

    void SetSerial(const std::array<std::string, 2> &serial);
    std::array<std::string, 2> GetSerial() const;

    void SetInputImageSize(const cv::Size &input_image_size);
    cv::Size GetInputImageSize();

    void SetCameraMatrix(const std::array<cv::Mat, 2> &camera_matrix);
    std::array<cv::Mat, 2> GetCameraMatrix();

    void SetDistCoeffs(const std::array<cv::Mat, 2> &dist_coeffs);
    std::array<cv::Mat, 2> GetDistCoeffs();

    void SetR(const std::array<cv::Mat, 2> &R);
    std::array<cv::Mat, 2> GetR();

    void SetP(const std::array<cv::Mat, 2> &P);
    std::array<cv::Mat, 2> GetP();

    void SetQ(const cv::Mat &Q);
    cv::Mat GetQ();

    void SetValid(const std::array<cv::Rect, 2> &valid);
    std::array<cv::Rect, 2> GetValid();

    bool Write(const std::string filename);
    bool Read(const std::string filename);

  private:
    class Impl;
    std::shared_ptr<Impl> impl_;

    std::array<std::string, 2> serial;
    cv::Size inputImageSize;
    std::array<cv::Mat, 2> cameraMatrix;
    std::array<cv::Mat, 2> distCoeffs;
    std::array<cv::Mat, 2> R;
    std::array<cv::Mat, 2> P;
    cv::Mat Q;
    std::array<cv::Rect, 2> valid;
};
} // namespace retinify