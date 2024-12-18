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

#include <iostream>
#include <retinify/data.hpp>
#include <stdexcept>

class retinify::CalibrationData::Impl
{
  public:
    std::array<std::string, 2> serial{"", ""};
    cv::Size inputImageSize{1280, 720};
    std::array<cv::Mat, 2> cameraMatrix{cv::Mat::eye(3, 3, CV_64F), cv::Mat::eye(3, 3, CV_64F)};
    std::array<cv::Mat, 2> distCoeffs{cv::Mat::zeros(5, 1, CV_64F), cv::Mat::zeros(5, 1, CV_64F)};
    std::array<cv::Mat, 2> R{cv::Mat::eye(3, 3, CV_64F), cv::Mat::eye(3, 3, CV_64F)};
    std::array<cv::Mat, 2> P{cv::Mat::eye(3, 4, CV_64F), cv::Mat::eye(3, 4, CV_64F)};
    cv::Mat Q{cv::Mat::eye(4, 4, CV_64F)};
    std::array<cv::Rect, 2> valid{cv::Rect(), cv::Rect()};
};

retinify::CalibrationData::CalibrationData()
{
    this->impl_ = std::make_shared<Impl>();
    inputImageSize = cv::Size(1280, 720);
    serial = {"", ""};
    cameraMatrix = {cv::Mat::eye(3, 3, CV_64F), cv::Mat::eye(3, 3, CV_64F)};
    distCoeffs = {cv::Mat::zeros(5, 1, CV_64F), cv::Mat::zeros(5, 1, CV_64F)};
    R = {cv::Mat::eye(3, 3, CV_64F), cv::Mat::eye(3, 3, CV_64F)};
    P = {cv::Mat::eye(3, 4, CV_64F), cv::Mat::eye(3, 4, CV_64F)};
    Q = cv::Mat::eye(4, 4, CV_64F);
    valid = {cv::Rect(), cv::Rect()};
}

retinify::CalibrationData::~CalibrationData() = default;

void retinify::CalibrationData::SetInputImageSize(const cv::Size &input_image_size)
{
    this->inputImageSize = input_image_size;
}

cv::Size retinify::CalibrationData::GetInputImageSize()
{
    return this->inputImageSize;
}

void retinify::CalibrationData::SetSerial(const std::array<std::string, 2> &serial)
{
    this->serial = serial;
}

std::array<std::string, 2> retinify::CalibrationData::GetSerial() const
{
    return this->serial;
}

void retinify::CalibrationData::SetCameraMatrix(const std::array<cv::Mat, 2> &camera_matrix)
{
    this->cameraMatrix = camera_matrix;
}

std::array<cv::Mat, 2> retinify::CalibrationData::GetCameraMatrix()
{
    return this->cameraMatrix;
}

void retinify::CalibrationData::SetDistCoeffs(const std::array<cv::Mat, 2> &dist_coeffs)
{
    this->distCoeffs = dist_coeffs;
}

std::array<cv::Mat, 2> retinify::CalibrationData::GetDistCoeffs()
{
    return this->distCoeffs;
}

void retinify::CalibrationData::SetR(const std::array<cv::Mat, 2> &R)
{
    this->R = R;
}

std::array<cv::Mat, 2> retinify::CalibrationData::GetR()
{
    return this->R;
}

void retinify::CalibrationData::SetP(const std::array<cv::Mat, 2> &P)
{
    this->P = P;
}

std::array<cv::Mat, 2> retinify::CalibrationData::GetP()
{
    return this->P;
}

void retinify::CalibrationData::SetQ(const cv::Mat &Q)
{
    this->Q = Q;
}

cv::Mat retinify::CalibrationData::GetQ()
{
    return this->Q;
}

void retinify::CalibrationData::SetValid(const std::array<cv::Rect, 2> &valid)
{
    this->valid = valid;
}

std::array<cv::Rect, 2> retinify::CalibrationData::GetValid()
{
    return this->valid;
}

#define CONFIGURATION_INPUT_IMAGE_SIZE "CONFIGURATION_INPUT_IMAGE_SIZE"
#define CONFIGURATION_SCALE "CONFIGURATION_SCALE"
#define CONFIGURATION_PROCESSING_IMAGE_SIZE "CONFIGURATION_PROCESSING_IMAGE_SIZE"
#define CONFIGURATION_SERIAL_0 "CONFIGURATION_SERIAL_0"
#define CONFIGURATION_SERIAL_1 "CONFIGURATION_SERIAL_1"
#define CONFIGURATION_CAMERA_MATRIX_0 "CONFIGURATION_CAMERA_MATRIX_0"
#define CONFIGURATION_CAMERA_MATRIX_1 "CONFIGURATION_CAMERA_MATRIX_1"
#define CONFIGURATION_DIST_COEFFS_0 "CONFIGURATION_DIST_COEFFS_0"
#define CONFIGURATION_DIST_COEFFS_1 "CONFIGURATION_DIST_COEFFS_1"
#define CONFIGURATION_R_1 "CONFIGURATION_R_1"
#define CONFIGURATION_R_2 "CONFIGURATION_R_2"
#define CONFIGURATION_P_1 "CONFIGURATION_P_1"
#define CONFIGURATION_P_2 "CONFIGURATION_P_2"
#define CONFIGURATION_Q "CONFIGURATION_Q"
#define CONFIGURATION_VALID_0 "CONFIGURATION_VALID_0"
#define CONFIGURATION_VALID_1 "CONFIGURATION_VALID_1"

bool retinify::CalibrationData::Write(std::string filename)
{
    try
    {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if (!fs.isOpened())
        {
            return false;
        }

        fs << CONFIGURATION_INPUT_IMAGE_SIZE << inputImageSize;
        fs << CONFIGURATION_SERIAL_0 << serial[0];
        fs << CONFIGURATION_SERIAL_1 << serial[1];
        fs << CONFIGURATION_CAMERA_MATRIX_0 << cameraMatrix[0];
        fs << CONFIGURATION_CAMERA_MATRIX_1 << cameraMatrix[1];
        fs << CONFIGURATION_DIST_COEFFS_0 << distCoeffs[0];
        fs << CONFIGURATION_DIST_COEFFS_1 << distCoeffs[1];
        fs << CONFIGURATION_R_1 << R[0];
        fs << CONFIGURATION_R_2 << R[1];
        fs << CONFIGURATION_P_1 << P[0];
        fs << CONFIGURATION_P_2 << P[1];
        fs << CONFIGURATION_Q << Q;
        fs << CONFIGURATION_VALID_0 << valid[0];
        fs << CONFIGURATION_VALID_1 << valid[1];

        fs.release();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error writing to file: " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool retinify::CalibrationData::Read(std::string filename)
{
    try
    {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            this->Write(filename);
            std::cout << "Creadted new calibration file" << std::endl;
        }

        fs[CONFIGURATION_INPUT_IMAGE_SIZE] >> inputImageSize;
        fs[CONFIGURATION_SERIAL_0] >> serial[0];
        fs[CONFIGURATION_SERIAL_1] >> serial[1];
        fs[CONFIGURATION_CAMERA_MATRIX_0] >> cameraMatrix[0];
        fs[CONFIGURATION_CAMERA_MATRIX_1] >> cameraMatrix[1];
        fs[CONFIGURATION_DIST_COEFFS_0] >> distCoeffs[0];
        fs[CONFIGURATION_DIST_COEFFS_1] >> distCoeffs[1];
        fs[CONFIGURATION_R_1] >> R[0];
        fs[CONFIGURATION_R_2] >> R[1];
        fs[CONFIGURATION_P_1] >> P[0];
        fs[CONFIGURATION_P_2] >> P[1];
        fs[CONFIGURATION_Q] >> Q;
        fs[CONFIGURATION_VALID_0] >> valid[0];
        fs[CONFIGURATION_VALID_1] >> valid[1];

        fs.release();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error reading from file: " << e.what() << std::endl;
        return false;
    }
    return true;
}
