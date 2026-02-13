// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/imgproc.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace retinify
{
TEST(ImgprocTest, ResizeSupportsOneChannel)
{
    constexpr std::size_t kSrcWidth = 5;
    constexpr std::size_t kSrcHeight = 4;
    constexpr std::size_t kDstWidth = 7;
    constexpr std::size_t kDstHeight = 6;
    constexpr std::size_t kChannels = 1;
    constexpr std::size_t kSrcStrideBytes = kSrcWidth * kChannels + 3;
    constexpr std::size_t kDstStrideBytes = kDstWidth * kChannels + 5;
    constexpr std::uint8_t kPaddingValue = 0xAB;

    std::vector<std::uint8_t> src(kSrcHeight * kSrcStrideBytes, kPaddingValue);
    std::vector<std::uint8_t> dst(kDstHeight * kDstStrideBytes, kPaddingValue);

    for (std::size_t y = 0; y < kSrcHeight; ++y)
    {
        std::uint8_t *row = src.data() + y * kSrcStrideBytes;
        for (std::size_t x = 0; x < kSrcWidth; ++x)
        {
            row[x] = static_cast<std::uint8_t>((y * 37 + x * 11 + 13) % 256);
        }
    }

    const Status status = Resize(src.data(), kSrcStrideBytes, dst.data(), kDstStrideBytes, kSrcWidth, kSrcHeight, kDstWidth, kDstHeight, kChannels);
    ASSERT_TRUE(status.IsOK());

    const cv::Mat srcView(static_cast<int>(kSrcHeight), static_cast<int>(kSrcWidth), CV_8UC1, src.data(), kSrcStrideBytes);
    cv::Mat expected;
    cv::resize(srcView, expected, cv::Size(static_cast<int>(kDstWidth), static_cast<int>(kDstHeight)), 0.0, 0.0, cv::INTER_LINEAR);

    for (std::size_t y = 0; y < kDstHeight; ++y)
    {
        const std::uint8_t *row = dst.data() + y * kDstStrideBytes;
        for (std::size_t x = 0; x < kDstWidth; ++x)
        {
            const int diff = std::abs(static_cast<int>(row[x]) - static_cast<int>(expected.at<std::uint8_t>(static_cast<int>(y), static_cast<int>(x))));
            EXPECT_LE(diff, 1) << "x=" << x << ", y=" << y;
        }
        for (std::size_t x = kDstWidth; x < kDstStrideBytes; ++x)
        {
            EXPECT_EQ(row[x], kPaddingValue);
        }
    }
}

TEST(ImgprocTest, ResizeSupportsThreeChannels)
{
    constexpr std::size_t kSrcWidth = 4;
    constexpr std::size_t kSrcHeight = 3;
    constexpr std::size_t kDstWidth = 6;
    constexpr std::size_t kDstHeight = 5;
    constexpr std::size_t kChannels = 3;
    constexpr std::size_t kSrcStrideBytes = kSrcWidth * kChannels + 7;
    constexpr std::size_t kDstStrideBytes = kDstWidth * kChannels + 9;
    constexpr std::uint8_t kPaddingValue = 0xC6;

    std::vector<std::uint8_t> src(kSrcHeight * kSrcStrideBytes, kPaddingValue);
    std::vector<std::uint8_t> dst(kDstHeight * kDstStrideBytes, kPaddingValue);

    for (std::size_t y = 0; y < kSrcHeight; ++y)
    {
        std::uint8_t *row = src.data() + y * kSrcStrideBytes;
        for (std::size_t x = 0; x < kSrcWidth; ++x)
        {
            for (std::size_t c = 0; c < kChannels; ++c)
            {
                row[x * kChannels + c] = static_cast<std::uint8_t>((y * 41 + x * 19 + c * 67 + 17) % 256);
            }
        }
    }

    const Status status = Resize(src.data(), kSrcStrideBytes, dst.data(), kDstStrideBytes, kSrcWidth, kSrcHeight, kDstWidth, kDstHeight, kChannels);
    ASSERT_TRUE(status.IsOK());

    const cv::Mat srcView(static_cast<int>(kSrcHeight), static_cast<int>(kSrcWidth), CV_8UC3, src.data(), kSrcStrideBytes);
    cv::Mat expected;
    cv::resize(srcView, expected, cv::Size(static_cast<int>(kDstWidth), static_cast<int>(kDstHeight)), 0.0, 0.0, cv::INTER_LINEAR);

    for (std::size_t y = 0; y < kDstHeight; ++y)
    {
        const std::uint8_t *row = dst.data() + y * kDstStrideBytes;
        for (std::size_t x = 0; x < kDstWidth; ++x)
        {
            const cv::Vec3b expectedPixel = expected.at<cv::Vec3b>(static_cast<int>(y), static_cast<int>(x));
            for (std::size_t c = 0; c < kChannels; ++c)
            {
                const int diff = std::abs(static_cast<int>(row[x * kChannels + c]) - static_cast<int>(expectedPixel[static_cast<int>(c)]));
                EXPECT_LE(diff, 1) << "x=" << x << ", y=" << y << ", c=" << c;
            }
        }
        for (std::size_t x = kDstWidth * kChannels; x < kDstStrideBytes; ++x)
        {
            EXPECT_EQ(row[x], kPaddingValue);
        }
    }
}

TEST(ImgprocTest, RemapSupportsOneChannel)
{
    constexpr std::size_t kWidth = 8;
    constexpr std::size_t kHeight = 6;
    constexpr std::size_t kChannels = 1;
    constexpr std::size_t kSrcStrideBytes = kWidth * kChannels + 4;
    constexpr std::size_t kDstStrideBytes = kWidth * kChannels + 6;
    constexpr std::size_t kMapStrideFloats = kWidth + 3;
    constexpr std::size_t kMapStrideBytes = kMapStrideFloats * sizeof(float);
    constexpr std::uint8_t kPaddingValue = 0xBD;

    std::vector<std::uint8_t> src(kHeight * kSrcStrideBytes, kPaddingValue);
    std::vector<std::uint8_t> dst(kHeight * kDstStrideBytes, kPaddingValue);
    std::vector<float> mapX(kHeight * kMapStrideFloats, -1.0F);
    std::vector<float> mapY(kHeight * kMapStrideFloats, -1.0F);

    for (std::size_t y = 0; y < kHeight; ++y)
    {
        std::uint8_t *srcRow = src.data() + y * kSrcStrideBytes;
        float *mapXRow = mapX.data() + y * kMapStrideFloats;
        float *mapYRow = mapY.data() + y * kMapStrideFloats;
        for (std::size_t x = 0; x < kWidth; ++x)
        {
            srcRow[x] = static_cast<std::uint8_t>((x * 19 + y * 23) % 256);
            mapXRow[x] = static_cast<float>(x) + 0.35F;
            mapYRow[x] = static_cast<float>(y) - 0.2F;
        }
    }

    const Status status = Remap(src.data(), kSrcStrideBytes, dst.data(), kDstStrideBytes, mapX.data(), kMapStrideBytes, mapY.data(), kMapStrideBytes, kWidth, kHeight, kChannels);
    ASSERT_TRUE(status.IsOK());

    const cv::Mat srcView(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_8UC1, src.data(), kSrcStrideBytes);
    const cv::Mat mapXView(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_32FC1, mapX.data(), kMapStrideBytes);
    const cv::Mat mapYView(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_32FC1, mapY.data(), kMapStrideBytes);
    cv::Mat expected;
    cv::remap(srcView, expected, mapXView, mapYView, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

    for (std::size_t y = 0; y < kHeight; ++y)
    {
        const std::uint8_t *row = dst.data() + y * kDstStrideBytes;
        for (std::size_t x = 0; x < kWidth; ++x)
        {
            const int diff = std::abs(static_cast<int>(row[x]) - static_cast<int>(expected.at<std::uint8_t>(static_cast<int>(y), static_cast<int>(x))));
            EXPECT_LE(diff, 3) << "x=" << x << ", y=" << y;
        }
        for (std::size_t x = kWidth; x < kDstStrideBytes; ++x)
        {
            EXPECT_EQ(row[x], kPaddingValue);
        }
    }
}

TEST(ImgprocTest, RemapSupportsThreeChannels)
{
    constexpr std::size_t kWidth = 7;
    constexpr std::size_t kHeight = 5;
    constexpr std::size_t kChannels = 3;
    constexpr std::size_t kSrcStrideBytes = kWidth * kChannels + 5;
    constexpr std::size_t kDstStrideBytes = kWidth * kChannels + 7;
    constexpr std::size_t kMapStrideFloats = kWidth + 2;
    constexpr std::size_t kMapStrideBytes = kMapStrideFloats * sizeof(float);
    constexpr std::uint8_t kPaddingValue = 0xE7;

    std::vector<std::uint8_t> src(kHeight * kSrcStrideBytes, kPaddingValue);
    std::vector<std::uint8_t> dst(kHeight * kDstStrideBytes, kPaddingValue);
    std::vector<float> mapX(kHeight * kMapStrideFloats, -1.0F);
    std::vector<float> mapY(kHeight * kMapStrideFloats, -1.0F);

    for (std::size_t y = 0; y < kHeight; ++y)
    {
        std::uint8_t *srcRow = src.data() + y * kSrcStrideBytes;
        float *mapXRow = mapX.data() + y * kMapStrideFloats;
        float *mapYRow = mapY.data() + y * kMapStrideFloats;
        for (std::size_t x = 0; x < kWidth; ++x)
        {
            for (std::size_t c = 0; c < kChannels; ++c)
            {
                srcRow[x * kChannels + c] = static_cast<std::uint8_t>((x * 17 + y * 29 + c * 37) % 256);
            }
            mapXRow[x] = static_cast<float>(x) + 0.2F;
            mapYRow[x] = static_cast<float>(y) + 0.3F;
        }
    }

    const Status status = Remap(src.data(), kSrcStrideBytes, dst.data(), kDstStrideBytes, mapX.data(), kMapStrideBytes, mapY.data(), kMapStrideBytes, kWidth, kHeight, kChannels);
    ASSERT_TRUE(status.IsOK());

    const cv::Mat srcView(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_8UC3, src.data(), kSrcStrideBytes);
    const cv::Mat mapXView(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_32FC1, mapX.data(), kMapStrideBytes);
    const cv::Mat mapYView(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_32FC1, mapY.data(), kMapStrideBytes);
    cv::Mat expected;
    cv::remap(srcView, expected, mapXView, mapYView, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

    for (std::size_t y = 0; y < kHeight; ++y)
    {
        const std::uint8_t *row = dst.data() + y * kDstStrideBytes;
        for (std::size_t x = 0; x < kWidth; ++x)
        {
            const cv::Vec3b expectedPixel = expected.at<cv::Vec3b>(static_cast<int>(y), static_cast<int>(x));
            for (std::size_t c = 0; c < kChannels; ++c)
            {
                const int diff = std::abs(static_cast<int>(row[x * kChannels + c]) - static_cast<int>(expectedPixel[static_cast<int>(c)]));
                EXPECT_LE(diff, 3) << "x=" << x << ", y=" << y << ", c=" << c;
            }
        }
        for (std::size_t x = kWidth * kChannels; x < kDstStrideBytes; ++x)
        {
            EXPECT_EQ(row[x], kPaddingValue);
        }
    }
}

TEST(ImgprocTest, ResizeAndRemapRejectUnsupportedChannels)
{
    std::vector<std::uint8_t> srcResize(4, 1U);
    std::vector<std::uint8_t> dstResize(4, 0U);
    const Status resizeStatus = Resize(srcResize.data(), 2, dstResize.data(), 2, 2, 1, 2, 1, 2);
    EXPECT_FALSE(resizeStatus.IsOK());
    EXPECT_EQ(resizeStatus.Category(), StatusCategory::USER);
    EXPECT_EQ(resizeStatus.Code(), StatusCode::INVALID_ARGUMENT);

    std::vector<std::uint8_t> srcRemap(4, 0U);
    std::vector<std::uint8_t> dstRemap(4, 0U);
    std::vector<float> mapX(2, 0.0F);
    std::vector<float> mapY(2, 0.0F);
    const Status remapStatus = Remap(srcRemap.data(), 2, dstRemap.data(), 2, mapX.data(), 2 * sizeof(float), mapY.data(), 2 * sizeof(float), 2, 1, 2);
    EXPECT_FALSE(remapStatus.IsOK());
    EXPECT_EQ(remapStatus.Category(), StatusCategory::USER);
    EXPECT_EQ(remapStatus.Code(), StatusCode::INVALID_ARGUMENT);
}
} // namespace retinify
