// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/log.hpp"
#include "retinify/status.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace retinify
{
class PipelineImageBuffer
{
  public:
    PipelineImageBuffer() noexcept = default;
    ~PipelineImageBuffer() noexcept = default;
    PipelineImageBuffer(const PipelineImageBuffer &) = delete;
    auto operator=(const PipelineImageBuffer &) noexcept -> PipelineImageBuffer & = delete;
    PipelineImageBuffer(PipelineImageBuffer &&) noexcept = delete;
    auto operator=(PipelineImageBuffer &&) noexcept -> PipelineImageBuffer & = delete;

    [[nodiscard]] auto Resize(std::size_t imageHeight, std::size_t imageWidth) -> Status
    {
        if (imageHeight == 0 || imageWidth == 0)
        {
            LogError("Height and width must be greater than zero.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        if (imageHeight > 720 || imageWidth > 1280)
        {
            LogError("Height and width exceed maximum allowed dimensions of 720x1280.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        imageHeight_ = imageHeight;
        imageWidth_ = imageWidth;
        imageStride_ = imageWidth * sizeof(float);

        return Status{};
    }

    [[nodiscard]] auto Upload(const std::uint8_t *imageData, const std::size_t imageStride) noexcept -> Status
    {
        if (imageData == nullptr)
        {
            LogError("Image data is nullptr.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        if (imageStride == 0)
        {
            LogError("Invalid imageStride (zero).");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        if (imageHeight_ == 0 || imageWidth_ == 0)
        {
            LogError("Buffer not resized.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(imageHeight_); ++row)
        {
            const std::uint8_t *srcRow = imageData + static_cast<std::size_t>(row) * imageStride;
            float *dstRow = buffer_.data() + static_cast<std::size_t>(row) * imageWidth_;

#ifdef _OPENMP
#pragma omp simd
#endif
            for (std::size_t col = 0; col < imageWidth_; ++col)
            {
                dstRow[col] = static_cast<float>(srcRow[col]);
            }
        }

        return Status{};
    }

    [[nodiscard]] auto Data() const noexcept -> const float *
    {
        return buffer_.data();
    }

    [[nodiscard]] auto Stride() const noexcept -> std::size_t
    {
        return imageStride_;
    }

  private:
    static constexpr std::size_t BufferSize = 720 * 1280 * 1;
    alignas(64) std::array<float, BufferSize> buffer_{};
    std::size_t imageHeight_{0};
    std::size_t imageWidth_{0};
    std::size_t imageStride_{0};
};
} // namespace retinify