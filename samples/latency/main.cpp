// Copyright (c) 2025 Sensui Yagi. All rights reserved.

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <retinify/retinify.hpp>

int main()
{
    retinify::SetLogLevel(retinify::LogLevel::INFO);
    retinify::Pipeline pipeline;

    (void)pipeline.Initialize(320, 640);

    cv::Mat img0 = cv::Mat::zeros(320, 640, CV_32FC3);
    cv::Mat img1 = cv::Mat::zeros(320, 640, CV_32FC3);
    cv::Mat disp = cv::Mat::zeros(320, 640, CV_32FC1);

    std::vector<double> latencies;
    latencies.reserve(1000);
    for (int i = 0; i < 1000; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto status = pipeline.Forward(img0.ptr(), img0.step, img1.ptr(), img1.step, disp.ptr(), disp.step);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        latencies.push_back(duration.count() / 1000.0); // Convert to milliseconds
        retinify::LogInfo("Iteration " + std::to_string(i) + ": " + std::to_string(latencies.back()) + " ms");
    }
    std::sort(latencies.begin(), latencies.end());
    double median = latencies[latencies.size() / 2];
    retinify::LogInfo("Median latency: " + std::to_string(median) + " ms");

    return 0;
}