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

#include <retinify/config.h>
#include <retinify/core.hpp>

std::filesystem::path retinify::GetShareDirPath()
{
    return std::filesystem::path(RETINIFY_SHARE_DIR);
}

std::vector<float> retinify::CVMatToFloatVector(const cv::Mat &src, const cv::Size &size)
{
    cv::Mat blob = cv::dnn::blobFromImage(src, 1.0, size, cv::Scalar(), true);
    return std::vector<float>(blob.begin<float>(), blob.end<float>());
}

cv::Mat retinify::DrawingLinesOnImage(const cv::Mat &src)
{
    cv::Mat dst = src.clone();

    for (int i = 0; i < dst.rows; i += 32)
    {
        cv::line(dst, cv::Point(0, i), cv::Point(dst.cols, i), cv::Scalar(0, 0, 255), 1);
    }

    return dst;
}

cv::Mat retinify::ColoringDisparity(const cv::Mat disparity, const int max, const bool rgb)
{
    if (disparity.empty())
    {
        return cv::Mat::zeros(480, 640, CV_8UC3);
    }

    cv::Mat show;

    // set disparity values greater than threshold to 0
    cv::Mat thresholded_disparity;
    cv::threshold(disparity, thresholded_disparity, max, 0, cv::THRESH_TOZERO_INV);

    // normalize disparity map
    cv::Mat normalized_disparity;
    thresholded_disparity.convertTo(normalized_disparity, CV_8U, 255.0 / max);

    // apply color map
    cv::applyColorMap(normalized_disparity, show, cv::COLORMAP_INFERNO);
    // cv::applyColorMap(normalized_disparity, show, cv::COLORMAP_JET);

    if (rgb)
    {
        cv::cvtColor(show, show, cv::COLOR_BGR2RGB);
    }

    return show;
}

cv::Mat retinify::computeQMatrix(const cv::Mat &cam0, const cv::Mat &cam1, double doffs, double baseline)
{
    double fx = cam0.at<double>(0, 0);
    double fy = cam0.at<double>(1, 1);
    double cx = cam0.at<double>(0, 2);
    double cy = cam0.at<double>(1, 2);

    cv::Mat Q =
        (cv::Mat_<double>(4, 4) << 1, 0, 0, -cx, 0, 1, 0, -cy, 0, 0, 0, fx, 0, 0, -1.0 / baseline, doffs / baseline);
    return Q;
}