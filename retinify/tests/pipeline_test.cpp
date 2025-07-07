// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/pipeline.hpp"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

namespace retinify
{
TEST(PipelineTest, Forward)
{
    const int height = 1280;
    const int width = 720;
    cv::Mat left = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat right = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat disp = cv::Mat::zeros(height, width, CV_32FC1);

    Pipeline pipeline;

    Status stInit = pipeline.Initialize(height, width);
    ASSERT_TRUE(stInit.IsOK()) << "Initialize Failed";

    Status stFwd = pipeline.Forward(left.ptr(), left.step, right.ptr(), right.step, disp.ptr(), disp.step);
    ASSERT_TRUE(stFwd.IsOK()) << "Forward Failed";
}
} // namespace retinify