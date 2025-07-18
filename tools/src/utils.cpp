// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/tools/utils.hpp"

#include <opencv2/imgproc.hpp>

namespace retinify::tools
{
cv::Mat ColorizeDisparity(const cv::Mat &disparity, int maxDisparity)
{
    if (disparity.empty())
    {
        return cv::Mat::zeros(disparity.size(), CV_32FC1);
    }

    cv::Mat coloredDisparity;

    // set disparity values greater than threshold to 0
    cv::Mat thresholdedDisparity;
    cv::threshold(disparity, thresholdedDisparity, maxDisparity, 0, cv::THRESH_TOZERO_INV);

    // normalize disparity map
    cv::Mat normalizedDisparity;
    thresholdedDisparity.convertTo(normalizedDisparity, CV_8UC1, 255.0 / maxDisparity);

    // apply color map
    cv::applyColorMap(normalizedDisparity, coloredDisparity, cv::COLORMAP_TURBO);
    return coloredDisparity;
}
} // namespace retinify::tools