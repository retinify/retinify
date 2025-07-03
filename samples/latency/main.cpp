// Copyright (c) 2025 Sensui Yagi. All rights reserved.

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <retinify/retinify.hpp>

int main()
{
    retinify::SetLogLevel(retinify::LogLevel::INFO);
    retinify::Pipeline pipeline;

    (void)pipeline.Initialize(720, 1280);
    cv::Mat img0 = cv::Mat::zeros(720, 1280, CV_32FC3);
    cv::Mat img1 = cv::Mat::zeros(720, 1280, CV_32FC3);
    cv::Mat disp = cv::Mat::zeros(720, 1280, CV_32FC1);

    std::vector<double> latencies;
    int num_frames = 10000;
    latencies.reserve(num_frames);
    for (int i = 0; i < num_frames; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto status = pipeline.Forward(img0.ptr(), img0.step, img1.ptr(), img1.step, disp.ptr(), disp.step);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        latencies.push_back(duration.count() / 1000.0); // Convert to milliseconds
        retinify::LogInfo(std::format("Frame {}: Latency = {:.3f} ms", i + 1, latencies.back()).c_str());
        }
        std::sort(latencies.begin(), latencies.end());
        double median = latencies[latencies.size() / 2];
        retinify::LogInfo(std::format("Median latency: {:.3f} ms", median).c_str());
    return 0;
}