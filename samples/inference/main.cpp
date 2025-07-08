// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/retinify.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

namespace retinify
{
cv::Mat ColoringDisparity(const cv::Mat disparity, const int maxDisparity)
{
    if (disparity.empty())
    {
        retinify::LogError("Disparity map is empty.");
        return cv::Mat();
    }

    cv::Mat coloredDisparity;

    // set disparity values greater than threshold to 0
    cv::Mat thresholdedDisparity;
    cv::threshold(disparity, thresholdedDisparity, maxDisparity, 0, cv::THRESH_TOZERO_INV);

    // normalize disparity map
    cv::Mat normalizedDisparity;
    thresholdedDisparity.convertTo(normalizedDisparity, CV_8UC1, 255.0 / maxDisparity);

    // apply color map
    cv::applyColorMap(normalizedDisparity, coloredDisparity, cv::COLORMAP_JET);
    return coloredDisparity;
}
} // namespace retinify

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <left_image_path> <right_image_path>" << std::endl;
        return 1;
    }

    std::string left_path = argv[1];
    std::string right_path = argv[2];

    retinify::SetLogLevel(retinify::LogLevel::INFO);
    retinify::Pipeline pipeline;

    auto statusInitialize = pipeline.Initialize(720, 1280);
    if (!statusInitialize.IsOK())
    {
        retinify::LogError("Failed to initialize the pipeline.");
        return 1;
    }

    cv::Mat leftImage = cv::imread(left_path);
    cv::Mat rightImage = cv::imread(right_path);
    if (leftImage.empty() || rightImage.empty())
    {
        retinify::LogError("Failed to load input images.");
        return 1;
    }

    cv::Size leftImageSize = leftImage.size();
    cv::Size rightImageSize = rightImage.size();
    if (leftImageSize != rightImageSize)
    {
        retinify::LogError("Input images must have the same size.");
        return 1;
    }

    cv::resize(leftImage, leftImage, cv::Size(1280, 720));
    cv::resize(rightImage, rightImage, cv::Size(1280, 720));

    leftImage.convertTo(leftImage, CV_32FC3);
    rightImage.convertTo(rightImage, CV_32FC3);
    cv::Mat disparity = cv::Mat{leftImage.size(), CV_32FC1};

    auto statusRun = pipeline.Run(leftImage.ptr(), leftImage.step[0], rightImage.ptr(), rightImage.step[0], disparity.ptr(), disparity.step[0]);
    if (!statusRun.IsOK())
    {
        retinify::LogError("Failed to run the pipeline.");
        return 1;
    }

    cv::resize(disparity, disparity, leftImageSize, 0, 0, cv::INTER_NEAREST);
    cv::imshow("disparity", retinify::ColoringDisparity(disparity, 128));
    cv::imwrite("disparity.png", retinify::ColoringDisparity(disparity, 128));
    cv::waitKey(0);

    return 0;
}