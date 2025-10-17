// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-eula

#include "retinify/retinify.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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

    cv::Mat leftImage = cv::imread(left_path);
    cv::Mat rightImage = cv::imread(right_path);
    if (leftImage.empty() || rightImage.empty())
    {
        retinify::LogError("Failed to load input images.");
        return 1;
    }

    cv::Mat disparity = cv::Mat::zeros(leftImage.size(), CV_32FC1);
    cv::Mat disparityColored = cv::Mat::zeros(leftImage.size(), CV_8UC3);

    auto statusInitialize = pipeline.Initialize(static_cast<std::uint32_t>(leftImage.cols), static_cast<std::uint32_t>(leftImage.rows));
    if (!statusInitialize.IsOK())
    {
        retinify::LogError("Failed to initialize the pipeline.");
        return 1;
    }

    auto statusExecute = pipeline.Execute(leftImage.ptr<std::uint8_t>(), leftImage.step[0], //
                                          rightImage.ptr<std::uint8_t>(), rightImage.step[0]);
    if (!statusExecute.IsOK())
    {
        retinify::LogError("Failed to run the pipeline.");
        return 1;
    }

    auto statusRetrieve = pipeline.RetrieveDisparity(disparity.ptr<float>(), disparity.step[0]);
    if (!statusRetrieve.IsOK())
    {
        retinify::LogError("Failed to retrieve the disparity map.");
        return 1;
    }

    auto statusColorize = retinify::ColorizeDisparity(disparity.ptr<float>(), disparity.step[0], disparityColored.ptr<std::uint8_t>(), disparityColored.step[0], disparity.cols, disparity.rows, 256.0F);
    if (!statusColorize.IsOK())
    {
        retinify::LogError("Failed to colorize the disparity map.");
        return 1;
    }

    cv::cvtColor(disparityColored, disparityColored, cv::COLOR_RGB2BGR);
    cv::imshow("disparity", disparityColored);
    cv::imwrite("disparity.png", disparityColored);
    cv::waitKey(0);

    return 0;
}
