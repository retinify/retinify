// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/pipeline.hpp"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

namespace retinify
{
TEST(PipelineTest, Forward)
{
    const std::uint32_t width = 1280;
    const std::uint32_t height = 720;
    cv::Mat left = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat right = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat disp = cv::Mat::zeros(height, width, CV_32FC1);

    Pipeline pipeline;

    Status stInit = pipeline.Initialize(width, height);
    ASSERT_TRUE(stInit.IsOK()) << "Initialize Failed";

    Status stFwd = pipeline.Run(left.ptr<std::uint8_t>(), left.step[0], right.ptr<std::uint8_t>(), right.step[0], disp.ptr<float>(), disp.step[0]);
    ASSERT_TRUE(stFwd.IsOK()) << "Forward Failed";
}
} // namespace retinify
