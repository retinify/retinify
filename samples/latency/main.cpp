// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/retinify.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

double BenchmarkPipeline(retinify::tools::Mode mode, const cv::Mat &img0, const cv::Mat &img1, cv::Mat &disp, int num_iters = 10000)
{
    retinify::tools::StereoMatchingPipeline pipeline;
    retinify::Status statusInitialize = pipeline.Initialize(mode);
    if (!statusInitialize.IsOK())
    {
        retinify::LogError("Pipeline initialization failed for mode.");
        return -1.0;
    }

    std::vector<double> latencies;
    latencies.reserve(num_iters);

    for (int i = 0; i < num_iters; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        retinify::Status statusRun = pipeline.Run(img0, img1, disp);
        if (!statusRun.IsOK())
        {
            retinify::LogError("Pipeline run failed for mode.");
            return -1.0;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double ms = static_cast<double>(duration.count()) / 1000.0;
        latencies.push_back(ms);
    }

    std::sort(latencies.begin(), latencies.end());
    double median = latencies[latencies.size() / 2];
    return median;
}

int main()
{
    constexpr int height = 720;
    constexpr int width = 1280;
    constexpr int num_iters = 10000;

    cv::Mat img0 = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat img1 = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat disp;

    retinify::SetLogLevel(retinify::LogLevel::INFO);

    const std::vector<retinify::tools::Mode> modes = {retinify::tools::Mode::FAST, retinify::tools::Mode::BALANCED, retinify::tools::Mode::ACCURATE};

    struct Result
    {
        std::string name;
        double median_ms;
        double fps;
    };
    std::vector<Result> results;
    results.reserve(modes.size());

    for (retinify::tools::Mode m : modes)
    {
        std::string mode_name;
        switch (m)
        {
        case retinify::tools::Mode::FAST:
            mode_name = "FAST";
            break;
        case retinify::tools::Mode::BALANCED:
            mode_name = "BALANCED";
            break;
        case retinify::tools::Mode::ACCURATE:
            mode_name = "ACCURATE";
            break;
        default:
            mode_name = "UNKNOWN";
            break;
        }

        retinify::LogInfo(cv::format("Running benchmark for mode: %s ...", mode_name.c_str()).c_str());

        double median_ms = BenchmarkPipeline(m, img0, img1, disp, num_iters);
        if (median_ms < 0.0)
        {
            retinify::LogError(cv::format("Benchmark failed for mode: %s", mode_name.c_str()).c_str());
            continue;
        }

        double fps = 1000.0 / median_ms;
        results.push_back({mode_name, median_ms, fps});
    }

    std::cout << std::endl << std::left << std::setw(10) << "Mode" << std::right << std::setw(15) << "Median (ms)" << std::setw(15) << "FPS" << std::endl << std::string(40, '-') << std::endl;

    for (auto const &r : results)
    {
        std::cout << std::left << std::setw(10) << r.name << std::right << std::setw(15) << std::fixed << std::setprecision(3) << r.median_ms << std::setw(15) << std::fixed << std::setprecision(1) << r.fps << std::endl;
    }

    return 0;
}
