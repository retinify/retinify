// Copyright (c) 2025 Sensui Yagi. All rights reserved.

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <retinify/retinify.hpp>

cv::Mat ColoringDisparity(const cv::Mat disparity, const int maxDisparity)
{
    if (disparity.empty())
    {
        retinify::LogError("Disparity map is empty.");
        return cv::Mat();
    }

    cv::Mat show;

    // set disparity values greater than threshold to 0
    cv::Mat thresholded_disparity;
    cv::threshold(disparity, thresholded_disparity, maxDisparity, 0, cv::THRESH_TOZERO_INV);

    // normalize disparity map
    cv::Mat normalized_disparity;
    thresholded_disparity.convertTo(normalized_disparity, CV_8UC1, 255.0 / maxDisparity);

    // apply color map
    cv::applyColorMap(normalized_disparity, show, cv::COLORMAP_JET);
    return show;
}

namespace retinify
{
void LogStatus(const retinify::Status &status)
{
    switch (status.Category())
    {
    case retinify::StatusCategory::NONE:
        std::cout << "NONE";
        break;
    case retinify::StatusCategory::RETINIFY:
        std::cout << "RETINIFY";
        break;
    case retinify::StatusCategory::SYSTEM:
        std::cout << "SYSTEM";
        break;
    case retinify::StatusCategory::CUDA:
        std::cout << "CUDA";
        break;
    case retinify::StatusCategory::USER:
        std::cout << "USER";
        break;
    default:
        std::cout << "UNKNOWN";
        break;
    }

    std::cout << " - Code: ";
    switch (status.Code())
    {
    case retinify::StatusCode::OK:
        std::cout << "OK";
        break;
    case retinify::StatusCode::FAIL:
        std::cout << "FAIL";
        break;
    case retinify::StatusCode::INVALID_ARGUMENT:
        std::cout << "INVALID_ARGUMENT";
        break;
    case retinify::StatusCode::NOT_ALLOCATED:
        std::cout << "NOT_ALLOCATED";
        break;
    case retinify::StatusCode::NULL_POINTER:
        std::cout << "NULL_POINTER";
        break;
    default:
        std::cout << static_cast<int>(status.Code());
        break;
    }
    std::cout << std::endl;
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

    retinify::SetLogLevel(retinify::LogLevel::DEBUG);
    retinify::Pipeline pipeline;

    (void)pipeline.Initialize(320, 640);

    cv::Mat img0 = cv::imread(left_path);
    cv::Mat img1 = cv::imread(right_path);
    cv::resize(img0, img0, cv::Size(640, 320));
    cv::resize(img1, img1, cv::Size(640, 320));
    img0.convertTo(img0, CV_32FC3);
    img1.convertTo(img1, CV_32FC3);
    cv::Mat disp = cv::Mat{img0.size(), CV_32FC1};

    (void)pipeline.Forward(img0.ptr(), img0.step, img1.ptr(), img1.step, disp.ptr(), disp.step);

    cv::Mat colored_disp = ColoringDisparity(disp, 128);
    cv::imshow("show", colored_disp);

    retinify::LogInfo("Inference Done.");

    cv::waitKey(0);

    return 0;
}