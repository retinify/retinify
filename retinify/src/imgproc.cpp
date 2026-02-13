// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/imgproc.hpp"
#include "retinify/logging.hpp"

#include <algorithm>
#include <cmath>

namespace retinify
{
auto Resize(const std::uint8_t *src, std::size_t srcStride, std::uint8_t *dst, std::size_t dstStride, std::size_t srcWidth, std::size_t srcHeight, std::size_t dstWidth, std::size_t dstHeight, std::size_t channels) noexcept -> Status
{
    if (src == nullptr || dst == nullptr)
    {
        LogError("source and destination pointers must not be null.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    if (srcWidth == 0U || srcHeight == 0U || dstWidth == 0U || dstHeight == 0U)
    {
        LogError("source and destination dimensions must be greater than zero.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    if (channels != 1U && channels != 3U)
    {
        LogError("channels must be 1 or 3.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    const std::size_t requiredSrcStride = srcWidth * channels * sizeof(std::uint8_t);
    if (srcStride < requiredSrcStride)
    {
        LogError("src stride is too small for the given source width and channels.");
        LogStrideError(srcStride, requiredSrcStride);
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    const std::size_t requiredDstStride = dstWidth * channels * sizeof(std::uint8_t);
    if (dstStride < requiredDstStride)
    {
        LogError("dst stride is too small for the given destination width and channels.");
        LogStrideError(dstStride, requiredDstStride);
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    const double scaleX = static_cast<double>(srcWidth) / static_cast<double>(dstWidth);
    const double scaleY = static_cast<double>(srcHeight) / static_cast<double>(dstHeight);
    const long long maxSrcX = static_cast<long long>(srcWidth - 1U);
    const long long maxSrcY = static_cast<long long>(srcHeight - 1U);

    for (std::size_t dstY = 0; dstY < dstHeight; ++dstY)
    {
        auto *dstRow = dst + dstY * dstStride;
        const double srcY = (static_cast<double>(dstY) + 0.5) * scaleY - 0.5;
        const long long srcY0Raw = static_cast<long long>(std::floor(srcY));
        const long long srcY1Raw = srcY0Raw + 1;
        const std::size_t srcY0 = static_cast<std::size_t>(std::clamp(srcY0Raw, 0LL, maxSrcY));
        const std::size_t srcY1 = static_cast<std::size_t>(std::clamp(srcY1Raw, 0LL, maxSrcY));
        const double weightY = srcY - static_cast<double>(srcY0Raw);

        const auto *srcRow0 = src + srcY0 * srcStride;
        const auto *srcRow1 = src + srcY1 * srcStride;

        for (std::size_t dstX = 0; dstX < dstWidth; ++dstX)
        {
            const double srcX = (static_cast<double>(dstX) + 0.5) * scaleX - 0.5;
            const long long srcX0Raw = static_cast<long long>(std::floor(srcX));
            const long long srcX1Raw = srcX0Raw + 1;
            const std::size_t srcX0 = static_cast<std::size_t>(std::clamp(srcX0Raw, 0LL, maxSrcX));
            const std::size_t srcX1 = static_cast<std::size_t>(std::clamp(srcX1Raw, 0LL, maxSrcX));
            const double weightX = srcX - static_cast<double>(srcX0Raw);

            for (std::size_t channel = 0; channel < channels; ++channel)
            {
                const std::size_t srcOffset00 = srcX0 * channels + channel;
                const std::size_t srcOffset10 = srcX1 * channels + channel;
                const int value00 = static_cast<int>(srcRow0[srcOffset00]);
                const int value10 = static_cast<int>(srcRow0[srcOffset10]);
                const int value01 = static_cast<int>(srcRow1[srcOffset00]);
                const int value11 = static_cast<int>(srcRow1[srcOffset10]);

                const double interpolated = (1.0 - weightX) * (1.0 - weightY) * static_cast<double>(value00) + //
                                            weightX * (1.0 - weightY) * static_cast<double>(value10) +         //
                                            (1.0 - weightX) * weightY * static_cast<double>(value01) +         //
                                            weightX * weightY * static_cast<double>(value11);
                const int rounded = static_cast<int>(std::lrint(interpolated));
                const int clamped = std::clamp(rounded, 0, 255);
                dstRow[dstX * channels + channel] = static_cast<std::uint8_t>(clamped);
            }
        }
    }

    return Status{};
}

auto Remap(const std::uint8_t *src, std::size_t srcStride, std::uint8_t *dst, std::size_t dstStride, const float *mapX, std::size_t mapXStride, const float *mapY, std::size_t mapYStride, std::size_t imageWidth, std::size_t imageHeight, std::size_t channels) noexcept -> Status
{
    if (src == nullptr || dst == nullptr || mapX == nullptr || mapY == nullptr)
    {
        LogError("source, destination, and map pointers must not be null.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    if (imageWidth == 0U || imageHeight == 0U)
    {
        LogError("image dimensions must be greater than zero.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    if (channels != 1U && channels != 3U)
    {
        LogError("channels must be 1 or 3.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    const std::size_t requiredImageStride = imageWidth * channels * sizeof(std::uint8_t);
    if (srcStride < requiredImageStride)
    {
        LogError("src stride is too small for the given image width and channels.");
        LogStrideError(srcStride, requiredImageStride);
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    if (dstStride < requiredImageStride)
    {
        LogError("dst stride is too small for the given image width and channels.");
        LogStrideError(dstStride, requiredImageStride);
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    const std::size_t requiredMapStride = imageWidth * sizeof(float);
    if (mapXStride < requiredMapStride)
    {
        LogError("mapX stride is too small for the given image width.");
        LogStrideError(mapXStride, requiredMapStride);
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    if (mapYStride < requiredMapStride)
    {
        LogError("mapY stride is too small for the given image width.");
        LogStrideError(mapYStride, requiredMapStride);
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    const auto *srcBytes = reinterpret_cast<const unsigned char *>(src);
    auto *dstBytes = reinterpret_cast<unsigned char *>(dst);
    const auto *mapXBytes = reinterpret_cast<const unsigned char *>(mapX);
    const auto *mapYBytes = reinterpret_cast<const unsigned char *>(mapY);

    const auto sample = [&](const std::uint8_t *row, long long x, std::size_t channel) noexcept -> int {
        if (row == nullptr || x < 0 || x >= static_cast<long long>(imageWidth))
        {
            return 0;
        }
        return static_cast<int>(row[static_cast<std::size_t>(x) * channels + channel]);
    };

    for (std::size_t y = 0; y < imageHeight; ++y)
    {
        const auto *mapXRow = reinterpret_cast<const float *>(mapXBytes + y * mapXStride);
        const auto *mapYRow = reinterpret_cast<const float *>(mapYBytes + y * mapYStride);
        auto *dstRow = reinterpret_cast<std::uint8_t *>(dstBytes + y * dstStride);

        for (std::size_t x = 0; x < imageWidth; ++x)
        {
            const double mappedX = static_cast<double>(mapXRow[x]);
            const double mappedY = static_cast<double>(mapYRow[x]);
            if (!std::isfinite(mappedX) || !std::isfinite(mappedY))
            {
                for (std::size_t channel = 0; channel < channels; ++channel)
                {
                    dstRow[x * channels + channel] = 0U;
                }
                continue;
            }

            const long long x0 = static_cast<long long>(std::floor(mappedX));
            const long long y0 = static_cast<long long>(std::floor(mappedY));
            const long long x1 = x0 + 1;
            const long long y1 = y0 + 1;
            const double weightX = mappedX - static_cast<double>(x0);
            const double weightY = mappedY - static_cast<double>(y0);

            const std::uint8_t *srcRow0 = nullptr;
            const std::uint8_t *srcRow1 = nullptr;
            if (y0 >= 0 && y0 < static_cast<long long>(imageHeight))
            {
                srcRow0 = reinterpret_cast<const std::uint8_t *>(srcBytes + static_cast<std::size_t>(y0) * srcStride);
            }
            if (y1 >= 0 && y1 < static_cast<long long>(imageHeight))
            {
                srcRow1 = reinterpret_cast<const std::uint8_t *>(srcBytes + static_cast<std::size_t>(y1) * srcStride);
            }

            for (std::size_t channel = 0; channel < channels; ++channel)
            {
                const int value00 = sample(srcRow0, x0, channel);
                const int value10 = sample(srcRow0, x1, channel);
                const int value01 = sample(srcRow1, x0, channel);
                const int value11 = sample(srcRow1, x1, channel);

                const double interpolated = (1.0 - weightX) * (1.0 - weightY) * static_cast<double>(value00) + //
                                            weightX * (1.0 - weightY) * static_cast<double>(value10) +         //
                                            (1.0 - weightX) * weightY * static_cast<double>(value01) +         //
                                            weightX * weightY * static_cast<double>(value11);
                const int rounded = static_cast<int>(std::lrint(interpolated));
                const int clamped = std::clamp(rounded, 0, 255);
                dstRow[x * channels + channel] = static_cast<std::uint8_t>(clamped);
            }
        }
    }

    return Status{};
}
} // namespace retinify
