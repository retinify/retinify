// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/status.hpp"

#include <cstddef>
#include <type_traits>

namespace retinify
{
/// @brief
/// This class provides an interface for running a stereo matching pipeline.
class RETINIFY_API Pipeline
{
  public:
    Pipeline() noexcept;
    ~Pipeline() noexcept;
    Pipeline(const Pipeline &) = delete;
    Pipeline &operator=(const Pipeline &) = delete;
    Pipeline(Pipeline &&) noexcept = delete;
    Pipeline &operator=(Pipeline &&) noexcept = delete;
    /// @brief
    /// Initializes the stereo matching pipeline with the specified image dimensions.
    /// @param imageHeight
    /// The height of the input images.
    /// @param imageWidth
    /// The width of the input images.
    /// @return
    /// A Status object indicating the success or failure of the initialization.
    [[nodiscard]] Status Initialize(const std::size_t imageHeight, const std::size_t imageWidth) noexcept;
    /// @brief
    /// Runs the stereo matching pipeline with the provided left and right image data.
    /// @param leftImageData
    /// The pointer to the left image data.
    /// @param leftImageStride
    /// The stride of the left image data in bytes.
    /// @param rightImageData
    /// The pointer to the right image data.
    /// @param rightImageStride
    /// The stride of the right image data in bytes.
    /// @param disparityData
    /// The pointer to the output disparity data.
    /// @param disparityStride
    /// The stride of the output disparity data in bytes.
    /// @return
    /// A Status object indicating the success or failure of the operation.
    [[nodiscard]] Status Run(const void *leftImageData, const std::size_t leftImageStride,   //
                             const void *rightImageData, const std::size_t rightImageStride, //
                             void *disparityData, const std::size_t disparityStride) const noexcept;

  private:
    class Impl;
    Impl *impl() noexcept;
    const Impl *impl() const noexcept;
    alignas(alignof(std::max_align_t)) unsigned char buffer_[512];
};
} // namespace retinify