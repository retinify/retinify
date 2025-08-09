// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mat.hpp"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

namespace retinify
{
class MatTest : public ::testing::Test
{
  protected:
    static constexpr std::size_t rows = 400;
    static constexpr std::size_t cols = 300;
    static constexpr std::size_t channels = 3;
    static constexpr std::size_t bytesPerElement = sizeof(float);

    Mat mat_;
    cv::Mat hostSrc_ = cv::Mat::eye(rows, cols *channels, CV_32F);
    cv::Mat hostDst_ = cv::Mat::zeros(rows, cols *channels, CV_32F);
};

TEST_F(MatTest, Allocate)
{
    Status status = mat_.Allocate(rows, cols, channels, bytesPerElement);
    ASSERT_TRUE(status.IsOK());

    ASSERT_EQ(mat_.Rows(), rows);
    ASSERT_EQ(mat_.Cols(), cols);
    ASSERT_EQ(mat_.Channels(), channels);
    ASSERT_EQ(mat_.BytesPerElement(), bytesPerElement);
    ASSERT_EQ(mat_.ElementCount(), rows * cols * channels);

    auto shape = mat_.Shape();
    ASSERT_EQ(shape[0], static_cast<int64_t>(1));
    ASSERT_EQ(shape[1], static_cast<int64_t>(rows));
    ASSERT_EQ(shape[2], static_cast<int64_t>(cols));
    ASSERT_EQ(shape[3], static_cast<int64_t>(channels));

    status = mat_.Free();
    ASSERT_TRUE(status.IsOK());
}

TEST_F(MatTest, UploadDownload)
{
    Status stAlloc = mat_.Allocate(rows, cols, channels, bytesPerElement);
    ASSERT_TRUE(stAlloc.IsOK());

    Status stUp = mat_.Upload(hostSrc_.ptr(), hostSrc_.step[0]);
    ASSERT_TRUE(stUp.IsOK());

    Status stWaitUp = mat_.Wait();
    ASSERT_TRUE(stWaitUp.IsOK());

    Status stDown = mat_.Download(hostDst_.ptr(), hostDst_.step[0]);
    ASSERT_TRUE(stDown.IsOK());

    Status stWaitDown = mat_.Wait();
    ASSERT_TRUE(stWaitDown.IsOK());

    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols * channels; ++j)
        {
            ASSERT_FLOAT_EQ(hostSrc_.at<float>(i, j), hostDst_.at<float>(i, j)) << "Mismatch at (" << i << ", " << j << ")";
        }
    }

    Status stFree = mat_.Free();
    ASSERT_TRUE(stFree.IsOK());
}

TEST_F(MatTest, UploadDownloadUInt8)
{
    constexpr std::size_t bytesPerElementUint8 = sizeof(std::uint8_t);

    cv::Mat hostSrc(rows, cols * channels, CV_8U);
    cv::randu(hostSrc, 0, 255);
    cv::Mat hostDst = cv::Mat::zeros(rows, cols * channels, CV_8U);

    Status stAlloc = mat_.Allocate(rows, cols, channels, bytesPerElementUint8);
    ASSERT_TRUE(stAlloc.IsOK());

    Status stUp = mat_.Upload(hostSrc.ptr(), hostSrc.step[0]);
    ASSERT_TRUE(stUp.IsOK());

    Status stWaitUp = mat_.Wait();
    ASSERT_TRUE(stWaitUp.IsOK());

    Status stDown = mat_.Download(hostDst.ptr(), hostDst.step[0]);
    ASSERT_TRUE(stDown.IsOK());

    Status stWaitDown = mat_.Wait();
    ASSERT_TRUE(stWaitDown.IsOK());

    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols * channels; ++j)
        {
            ASSERT_EQ(hostSrc.at<std::uint8_t>(i, j), hostDst.at<std::uint8_t>(i, j)) << "Mismatch at (" << i << ", " << j << ")";
        }
    }

    Status stFree = mat_.Free();
    ASSERT_TRUE(stFree.IsOK());
}
} // namespace retinify