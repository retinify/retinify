// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-EULA

#include "retinify/pipeline.hpp"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <vector>

namespace retinify
{
namespace
{
void InitializeAndRunPipeline(Pipeline &pipeline, std::uint32_t width, std::uint32_t height, PixelFormat pixelFormat, DepthMode depthMode)
{
    const int cvType = (pixelFormat == PixelFormat::RGB8) ? CV_8UC3 : CV_8UC1;
    cv::Mat left = cv::Mat::zeros(height, width, cvType);
    cv::Mat right = cv::Mat::zeros(height, width, cvType);
    cv::Mat disparity = cv::Mat::zeros(height, width, CV_32FC1);

    Status stInit = pipeline.Initialize(width, height, pixelFormat, depthMode);
    ASSERT_TRUE(stInit.IsOK()) << "Initialize Failed";

    Status stRun = pipeline.Run(left.ptr<std::uint8_t>(), left.step[0], right.ptr<std::uint8_t>(), right.step[0], disparity.ptr<float>(), disparity.step[0]);
    ASSERT_TRUE(stRun.IsOK()) << "Run Failed";
}
} // namespace

TEST(PipelineTest, RunGray)
{
    const std::uint32_t width = 1280;
    const std::uint32_t height = 720;
    cv::Mat left = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat right = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat disp = cv::Mat::zeros(height, width, CV_32FC1);

    Pipeline pipeline;

    Status stInit = pipeline.Initialize(width, height, PixelFormat::GRAY8, DepthMode::BALANCED);
    ASSERT_TRUE(stInit.IsOK()) << "Initialize Failed";

    Status stRun = pipeline.Run(left.ptr<std::uint8_t>(), left.step[0], right.ptr<std::uint8_t>(), right.step[0], disp.ptr<float>(), disp.step[0]);
    ASSERT_TRUE(stRun.IsOK()) << "Run Failed";
}

TEST(PipelineTest, RetrieveOutputsSuccess)
{
    const std::uint32_t width = 1280;
    const std::uint32_t height = 720;

    Pipeline pipeline;
    InitializeAndRunPipeline(pipeline, width, height, PixelFormat::RGB8, DepthMode::BALANCED);

    cv::Mat rectifiedLeft(height, width, CV_8UC3);
    Status stRetrieveLeft = pipeline.RetrieveRectifiedLeftImage(rectifiedLeft.ptr<std::uint8_t>(), rectifiedLeft.step[0]);
    ASSERT_TRUE(stRetrieveLeft.IsOK()) << "RetrieveRectifiedLeftImage Failed";

    cv::Mat rectifiedRight(height, width, CV_8UC3);
    Status stRetrieveRight = pipeline.RetrieveRectifiedRightImage(rectifiedRight.ptr<std::uint8_t>(), rectifiedRight.step[0]);
    ASSERT_TRUE(stRetrieveRight.IsOK()) << "RetrieveRectifiedRightImage Failed";

    cv::Mat rectifiedLeftPair(height, width, CV_8UC3);
    cv::Mat rectifiedRightPair(height, width, CV_8UC3);
    Status stRetrievePair = pipeline.RetrieveRectifiedImages(rectifiedLeftPair.ptr<std::uint8_t>(), rectifiedLeftPair.step[0], //
                                                             rectifiedRightPair.ptr<std::uint8_t>(), rectifiedRightPair.step[0]);
    ASSERT_TRUE(stRetrievePair.IsOK()) << "RetrieveRectifiedImages Failed";

    cv::Mat retrievedDisparity(height, width, CV_32FC1);
    Status stRetrieveDisparity = pipeline.RetrieveDisparity(retrievedDisparity.ptr<float>(), retrievedDisparity.step[0]);
    ASSERT_TRUE(stRetrieveDisparity.IsOK()) << "RetrieveDisparity Failed";

    std::vector<float> pointCloud(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U);
    const std::size_t pointCloudStride = static_cast<std::size_t>(width) * 3U * sizeof(float);
    Status stRetrievePointCloud = pipeline.RetrievePointCloud(pointCloud.data(), pointCloudStride);
    ASSERT_TRUE(stRetrievePointCloud.IsOK()) << "RetrievePointCloud Failed";
}

TEST(PipelineTest, RetrieveOutputsGraySuccess)
{
    const std::uint32_t width = 640;
    const std::uint32_t height = 480;

    Pipeline pipeline;
    InitializeAndRunPipeline(pipeline, width, height, PixelFormat::GRAY8, DepthMode::BALANCED);

    cv::Mat rectifiedLeft(height, width, CV_8UC1);
    Status stRetrieveLeft = pipeline.RetrieveRectifiedLeftImage(rectifiedLeft.ptr<std::uint8_t>(), rectifiedLeft.step[0]);
    ASSERT_TRUE(stRetrieveLeft.IsOK()) << "RetrieveRectifiedLeftImage Failed";

    cv::Mat rectifiedRight(height, width, CV_8UC1);
    Status stRetrieveRight = pipeline.RetrieveRectifiedRightImage(rectifiedRight.ptr<std::uint8_t>(), rectifiedRight.step[0]);
    ASSERT_TRUE(stRetrieveRight.IsOK()) << "RetrieveRectifiedRightImage Failed";

    cv::Mat rectifiedLeftPair(height, width, CV_8UC1);
    cv::Mat rectifiedRightPair(height, width, CV_8UC1);
    Status stRetrievePair = pipeline.RetrieveRectifiedImages(rectifiedLeftPair.ptr<std::uint8_t>(), rectifiedLeftPair.step[0], //
                                                             rectifiedRightPair.ptr<std::uint8_t>(), rectifiedRightPair.step[0]);
    ASSERT_TRUE(stRetrievePair.IsOK()) << "RetrieveRectifiedImages Failed";

    cv::Mat retrievedDisparity(height, width, CV_32FC1);
    Status stRetrieveDisparity = pipeline.RetrieveDisparity(retrievedDisparity.ptr<float>(), retrievedDisparity.step[0]);
    ASSERT_TRUE(stRetrieveDisparity.IsOK()) << "RetrieveDisparity Failed";
}

TEST(PipelineTest, RetrieveRectifiedImagesInvalidStride)
{
    const std::uint32_t width = 640;
    const std::uint32_t height = 480;

    Pipeline pipeline;
    InitializeAndRunPipeline(pipeline, width, height, PixelFormat::RGB8, DepthMode::BALANCED);

    std::vector<std::uint8_t> left(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U);
    std::vector<std::uint8_t> right(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U);
    const std::size_t requiredStride = static_cast<std::size_t>(width) * 3U * sizeof(std::uint8_t);
    const std::size_t invalidStride = requiredStride - 1U;

    Status status = pipeline.RetrieveRectifiedImages(left.data(), invalidStride, right.data(), invalidStride);
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::USER);
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST(PipelineTest, RetrieveRectifiedImagesNullBuffer)
{
    const std::uint32_t width = 640;
    const std::uint32_t height = 480;

    Pipeline pipeline;
    InitializeAndRunPipeline(pipeline, width, height, PixelFormat::RGB8, DepthMode::BALANCED);

    std::vector<std::uint8_t> right(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U);
    const std::size_t stride = static_cast<std::size_t>(width) * 3U * sizeof(std::uint8_t);

    Status status = pipeline.RetrieveRectifiedImages(nullptr, stride, right.data(), stride);
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::USER);
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);

    status = pipeline.RetrieveRectifiedImages(right.data(), stride, nullptr, stride);
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::USER);
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST(PipelineTest, RetrieveRectifiedLeftImageNullBuffer)
{
    const std::uint32_t width = 640;
    const std::uint32_t height = 480;

    Pipeline pipeline;
    InitializeAndRunPipeline(pipeline, width, height, PixelFormat::RGB8, DepthMode::BALANCED);

    const std::size_t stride = static_cast<std::size_t>(width) * 3U * sizeof(std::uint8_t);
    Status status = pipeline.RetrieveRectifiedLeftImage(nullptr, stride);
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::USER);
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST(PipelineTest, RetrieveRectifiedLeftImageInvalidStride)
{
    const std::uint32_t width = 640;
    const std::uint32_t height = 480;

    Pipeline pipeline;
    InitializeAndRunPipeline(pipeline, width, height, PixelFormat::RGB8, DepthMode::BALANCED);

    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U);
    const std::size_t requiredStride = static_cast<std::size_t>(width) * 3U * sizeof(std::uint8_t);

    Status status = pipeline.RetrieveRectifiedLeftImage(buffer.data(), requiredStride - 1U);
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::USER);
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST(PipelineTest, RetrieveRectifiedRightImageInvalidStride)
{
    const std::uint32_t width = 640;
    const std::uint32_t height = 480;

    Pipeline pipeline;
    InitializeAndRunPipeline(pipeline, width, height, PixelFormat::RGB8, DepthMode::BALANCED);

    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U);
    const std::size_t requiredStride = static_cast<std::size_t>(width) * 3U * sizeof(std::uint8_t);

    Status status = pipeline.RetrieveRectifiedRightImage(buffer.data(), requiredStride - 1U);
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::USER);
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST(PipelineTest, RetrieveRectifiedRightImageNullBuffer)
{
    const std::uint32_t width = 640;
    const std::uint32_t height = 480;

    Pipeline pipeline;
    InitializeAndRunPipeline(pipeline, width, height, PixelFormat::RGB8, DepthMode::BALANCED);

    const std::size_t stride = static_cast<std::size_t>(width) * 3U * sizeof(std::uint8_t);
    Status status = pipeline.RetrieveRectifiedRightImage(nullptr, stride);
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::USER);
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST(PipelineTest, RetrieveDisparityNullBuffer)
{
    const std::uint32_t width = 640;
    const std::uint32_t height = 480;

    Pipeline pipeline;
    InitializeAndRunPipeline(pipeline, width, height, PixelFormat::RGB8, DepthMode::BALANCED);

    const std::size_t stride = static_cast<std::size_t>(width) * sizeof(float);
    Status status = pipeline.RetrieveDisparity(nullptr, stride);
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::USER);
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST(PipelineTest, RetrieveDisparityInvalidStride)
{
    const std::uint32_t width = 640;
    const std::uint32_t height = 480;

    Pipeline pipeline;
    InitializeAndRunPipeline(pipeline, width, height, PixelFormat::RGB8, DepthMode::BALANCED);

    std::vector<float> disparity(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    const std::size_t requiredStride = static_cast<std::size_t>(width) * sizeof(float);

    Status status = pipeline.RetrieveDisparity(disparity.data(), requiredStride - sizeof(float));
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::USER);
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST(PipelineTest, RetrievePointCloudNullBuffer)
{
    const std::uint32_t width = 640;
    const std::uint32_t height = 480;

    Pipeline pipeline;
    InitializeAndRunPipeline(pipeline, width, height, PixelFormat::RGB8, DepthMode::BALANCED);

    const std::size_t stride = static_cast<std::size_t>(width) * 3U * sizeof(float);
    Status status = pipeline.RetrievePointCloud(nullptr, stride);
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::USER);
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST(PipelineTest, RetrievePointCloudInvalidStride)
{
    const std::uint32_t width = 640;
    const std::uint32_t height = 480;

    Pipeline pipeline;
    InitializeAndRunPipeline(pipeline, width, height, PixelFormat::RGB8, DepthMode::BALANCED);

    std::vector<float> pointCloud(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U);
    const std::size_t requiredStride = static_cast<std::size_t>(width) * 3U * sizeof(float);

    Status status = pipeline.RetrievePointCloud(pointCloud.data(), requiredStride - sizeof(float));
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::USER);
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST(PipelineTest, RetrieveRectifiedLeftImageBeforeInitialize)
{
    Pipeline pipeline;
    std::vector<std::uint8_t> buffer(1U);

    Status status = pipeline.RetrieveRectifiedLeftImage(buffer.data(), buffer.size());
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::USER);
    EXPECT_EQ(status.Code(), StatusCode::FAIL);
}

TEST(PipelineTest, RetrieveRectifiedRightImageBeforeInitialize)
{
    Pipeline pipeline;
    std::vector<std::uint8_t> buffer(1U);

    Status status = pipeline.RetrieveRectifiedRightImage(buffer.data(), buffer.size());
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::USER);
    EXPECT_EQ(status.Code(), StatusCode::FAIL);
}

TEST(PipelineTest, RetrieveRectifiedImagesBeforeInitialize)
{
    Pipeline pipeline;
    std::vector<std::uint8_t> left(1U);
    std::vector<std::uint8_t> right(1U);

    Status status = pipeline.RetrieveRectifiedImages(left.data(), left.size(), right.data(), right.size());
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::USER);
    EXPECT_EQ(status.Code(), StatusCode::FAIL);
}

TEST(PipelineTest, RetrieveDisparityBeforeInitialize)
{
    Pipeline pipeline;
    std::vector<float> buffer(1U);

    Status status = pipeline.RetrieveDisparity(buffer.data(), buffer.size() * sizeof(float));
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::USER);
    EXPECT_EQ(status.Code(), StatusCode::FAIL);
}

TEST(PipelineTest, RetrievePointCloudBeforeInitialize)
{
    Pipeline pipeline;
    std::vector<float> buffer(3U);

    Status status = pipeline.RetrievePointCloud(buffer.data(), buffer.size() * sizeof(float));
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::USER);
    EXPECT_EQ(status.Code(), StatusCode::FAIL);
}

TEST(PipelineTest, RunRGB)
{
    const std::uint32_t width = 1280;
    const std::uint32_t height = 720;
    cv::Mat left = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat right = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat disp = cv::Mat::zeros(height, width, CV_32FC1);

    Pipeline pipeline;

    Status stInit = pipeline.Initialize(width, height, PixelFormat::RGB8, DepthMode::BALANCED);
    ASSERT_TRUE(stInit.IsOK()) << "Initialize Failed";

    Status stRun = pipeline.Run(left.ptr<std::uint8_t>(), left.step[0], right.ptr<std::uint8_t>(), right.step[0], disp.ptr<float>(), disp.step[0]);
    ASSERT_TRUE(stRun.IsOK()) << "Run Failed";
}

TEST(PipelineTest, RunLowResolution)
{
    const std::uint32_t width = 640;
    const std::uint32_t height = 480;
    cv::Mat left = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat right = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat disp = cv::Mat::zeros(height, width, CV_32FC1);

    Pipeline pipeline;

    Status stInit = pipeline.Initialize(width, height, PixelFormat::RGB8, DepthMode::ACCURATE);
    ASSERT_TRUE(stInit.IsOK()) << "Initialize Failed";

    Status stRun = pipeline.Run(left.ptr<std::uint8_t>(), left.step[0], right.ptr<std::uint8_t>(), right.step[0], disp.ptr<float>(), disp.step[0]);
    ASSERT_TRUE(stRun.IsOK()) << "Run Failed";
}

TEST(PipelineTest, RunAccurateHighResolution)
{
    const std::uint32_t width = 3840;
    const std::uint32_t height = 2160;
    cv::Mat left = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat right = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat disp = cv::Mat::zeros(height, width, CV_32FC1);

    Pipeline pipeline;

    Status stInit = pipeline.Initialize(width, height, PixelFormat::RGB8, DepthMode::ACCURATE);
    ASSERT_TRUE(stInit.IsOK()) << "Initialize Failed";

    Status stRun = pipeline.Run(left.ptr<std::uint8_t>(), left.step[0], right.ptr<std::uint8_t>(), right.step[0], disp.ptr<float>(), disp.step[0]);
    ASSERT_TRUE(stRun.IsOK()) << "Run Failed";
}
} // namespace retinify
