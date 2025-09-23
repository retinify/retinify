// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "stream.hpp"

#include "retinify/attributes.hpp"
#include "retinify/status.hpp"

#include <array>

#ifdef BUILD_WITH_TENSORRT
#include <cuda_runtime.h>
#endif

namespace retinify
{
enum class MatLocation
{
    UNKNOWN,
    DEVICE,
    HOST,
};

class RETINIFY_API Mat
{
  public:
    Mat() noexcept = default;
    ~Mat() noexcept;
    Mat(const Mat &) = delete;
    auto operator=(const Mat &) -> Mat & = delete;
    Mat(Mat &&other) noexcept = delete;
    auto operator=(Mat &&other) noexcept -> Mat & = delete;
    [[nodiscard]] auto Allocate(std::size_t rows, std::size_t cols, std::size_t channels, std::size_t bytesPerElement, MatLocation location) noexcept -> Status;
    [[nodiscard]] auto Free() noexcept -> Status;
    [[nodiscard]] auto Upload(const void *hostData, std::size_t hostStride, Stream &stream) const noexcept -> Status;
    [[nodiscard]] auto Download(void *hostData, std::size_t hostStride, Stream &stream) const noexcept -> Status;
    [[nodiscard]] auto Data() const noexcept -> void *;
    [[nodiscard]] auto Empty() const noexcept -> bool;
    [[nodiscard]] auto Rows() const noexcept -> std::size_t;
    [[nodiscard]] auto Cols() const noexcept -> std::size_t;
    [[nodiscard]] auto Channels() const noexcept -> std::size_t;
    [[nodiscard]] auto BytesPerElement() const noexcept -> std::size_t;
    [[nodiscard]] auto ElementCount() const noexcept -> std::size_t;
    [[nodiscard]] auto Stride() const noexcept -> std::size_t;
    [[nodiscard]] auto Shape() const noexcept -> std::array<int64_t, 4>;
    [[nodiscard]] auto Location() const noexcept -> MatLocation;

  private:
    std::size_t rows_{0};
    std::size_t cols_{0};
    std::size_t channels_{0};
    std::size_t bytesPerElement_{0};
    std::size_t deviceRows_{0};
    std::size_t deviceColumnsInBytes_{0};
    std::size_t deviceStride_{0};
    void *deviceData_{nullptr};
    MatLocation location_{MatLocation::UNKNOWN};
};
} // namespace retinify
