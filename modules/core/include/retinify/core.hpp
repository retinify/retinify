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
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <retinify/data.hpp>
#include <retinify/mutex.hpp>
#include <retinify/queue.hpp>
#include <retinify/singleton.hpp>
#include <retinify/thread.hpp>
#define RETINIFY_SHARE_DIR_PATH (retinify::GetShareDirPath())
#define RETINIFY_RESOURCE_DIR_PATH (retinify::GetShareDirPath() / "resources")
#define RETINIFY_DEFAULT_CALIBRATION_FILE_PATH (retinify::GetShareDirPath() / "resources" / "calibration.yml")
#define RETINIFY_GUI_STYLE_CSS_PATH (retinify::GetShareDirPath() / "resources" / "style.css")
#define RETINIFY_DEFALUT_MODEL_PATH (retinify::GetShareDirPath() / "models" / "model.onnx")
namespace retinify
{
std::filesystem::path GetShareDirPath();
std::vector<float> CVMatToFloatVector(const cv::Mat &src, const cv::Size &size);
cv::Mat DrawingLinesOnImage(const cv::Mat &src);
cv::Mat ColoringDisparity(const cv::Mat disparity, const int max = 256, const bool rgb = false);
cv::Mat computeQMatrix(const cv::Mat &cam0, const cv::Mat &cam1, double doffs, double baseline);
} // namespace retinify