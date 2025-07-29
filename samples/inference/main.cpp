// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/retinify.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

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
    retinify::tools::StereoMatchingPipeline pipeline;

    auto statusInitialize = pipeline.Initialize();
    if (!statusInitialize.IsOK())
    {
        retinify::LogError("Failed to initialize the pipeline.");
        return 1;
    }

    cv::Mat leftImage = cv::imread(left_path);
    cv::Mat rightImage = cv::imread(right_path);
    cv::Mat disparity;
    if (leftImage.empty() || rightImage.empty())
    {
        retinify::LogError("Failed to load input images.");
        return 1;
    }

    auto statusRun = pipeline.Run(leftImage, rightImage, disparity, 5.0F);
    if (!statusRun.IsOK())
    {
        retinify::LogError("Failed to run the pipeline.");
        return 1;
    }

    cv::imshow("disparity", retinify::tools::ColorizeDisparity(disparity, 256));
    cv::imwrite("disparity.png", retinify::tools::ColorizeDisparity(disparity, 256));
    cv::waitKey(0);

    return 0;
}