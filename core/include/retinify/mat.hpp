// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/define.hpp"
#include "retinify/status.hpp"

#include <array>
#include <cstddef>

namespace retinify
{
class RETINIFY_API Mat
{
  public:
    Mat() noexcept;
    ~Mat() noexcept;
    Mat(const Mat &) = delete;
    auto operator=(const Mat &) -> Mat & = delete;
    Mat(Mat &&other) noexcept = delete;
    auto operator=(Mat &&other) noexcept -> Mat & = delete;
    [[nodiscard]] auto Allocate(std::size_t rows, std::size_t cols, std::size_t channels, std::size_t bytesPerElement = sizeof(float)) noexcept -> Status;
    [[nodiscard]] auto Free() noexcept -> Status;
    [[nodiscard]] auto Upload(const void *hostData, std::size_t hostStride) const noexcept -> Status;
    [[nodiscard]] auto Download(void *hostData, std::size_t hostStride) const noexcept -> Status;
    [[nodiscard]] auto Wait() const noexcept -> Status;
    [[nodiscard]] auto Data() const noexcept -> void *;
    [[nodiscard]] auto Rows() const noexcept -> std::size_t;
    [[nodiscard]] auto Cols() const noexcept -> std::size_t;
    [[nodiscard]] auto Channels() const noexcept -> std::size_t;
    [[nodiscard]] auto BytesPerElement() const noexcept -> std::size_t;
    [[nodiscard]] auto ElementCount() const noexcept -> std::size_t;
    [[nodiscard]] auto Stride() const noexcept -> std::size_t;
    [[nodiscard]] auto Shape() const noexcept -> std::array<int64_t, 4>;

  private:
    class Impl;
    auto impl() noexcept -> Impl *;
    [[nodiscard]] auto impl() const noexcept -> const Impl *;
    static constexpr std::size_t BufferSize = 512;
    alignas(alignof(std::max_align_t)) std::array<unsigned char, BufferSize> buffer_{};
};
} // namespace retinify
