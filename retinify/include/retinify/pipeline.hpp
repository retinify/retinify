// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/status.hpp"

#include <array>
#include <cstddef>

namespace retinify
{
/// @brief
/// A `retinify::Pipeline` provides an interface for running a stereo matching.
class RETINIFY_API Pipeline
{
  public:
    Pipeline() noexcept;
    ~Pipeline() noexcept;
    Pipeline(const Pipeline &) = delete;
    auto operator=(const Pipeline &) noexcept -> Pipeline & = delete;
    Pipeline(Pipeline &&) noexcept = delete;
    auto operator=(Pipeline &&) noexcept -> Pipeline & = delete;

    /// @brief
    /// Initializes the stereo matching pipeline with the given image dimensions.
    /// @param imageHeight
    /// Height of the input images (in pixels).
    /// @param imageWidth
    /// Width of the input images (in pixels).
    /// @param mode
    /// The mode option for the stereo matching.
    /// @return
    /// A Status object indicating whether the initialization was successful.
    [[nodiscard]] auto Initialize(std::size_t imageHeight, std::size_t imageWidth, //
                                  Mode mode = Mode::ACCURATE) noexcept -> Status;

    /// @brief
    /// Executes the stereo matching pipeline using the given left and right image data.
    /// @param leftImageData
    /// Pointer to the left image data (8-bit rgb).
    /// @param leftImageStride
    /// Stride (in bytes) of a row in the left image.
    /// @param rightImageData
    /// Pointer to the right image data (8-bit rgb).
    /// @param rightImageStride
    /// Stride (in bytes) of a row in the right image.
    /// @param disparityData
    /// Pointer to the output buffer for disparity data (32-bit float).
    /// @param disparityStride
    /// Stride (in bytes) of a row in the output disparity data.
    /// @return
    /// A Status object indicating whether the operation was successful.
    [[nodiscard]] auto Run(const std::uint8_t *leftImageData, std::size_t leftImageStride,   //
                           const std::uint8_t *rightImageData, std::size_t rightImageStride, //
                           float *disparityData, std::size_t disparityStride) noexcept -> Status;

    /// @brief
    /// Executes the stereo matching pipeline using the given left and right image data with left-right consistency check.
    /// @param leftImageData
    /// Pointer to the left image data (8-bit rgb).
    /// @param leftImageStride
    /// Stride (in bytes) of a row in the left image.
    /// @param rightImageData
    /// Pointer to the right image data (8-bit rgb).
    /// @param rightImageStride
    /// Stride (in bytes) of a row in the right image.
    /// @param disparityData
    /// Pointer to the output buffer for disparity data (32-bit float).
    /// @param disparityStride
    /// Stride (in bytes) of a row in the output disparity data.
    /// @param maxDisparityDifference
    /// Maximum allowable disparity difference used in the left-right consistency check.
    /// A negative value disables the consistency check.
    /// @return
    /// A Status object indicating whether the operation was successful.
    [[nodiscard]] auto Run(const std::uint8_t *leftImageData, std::size_t leftImageStride,   //
                           const std::uint8_t *rightImageData, std::size_t rightImageStride, //
                           float *disparityData, std::size_t disparityStride,                //
                           const float maxDisparityDifference) noexcept -> Status;

  private:
    class Impl;
    auto impl() noexcept -> Impl *;
    [[nodiscard]] auto impl() const noexcept -> const Impl *;
    static constexpr std::size_t BufferSize = 2048;
    alignas(alignof(std::max_align_t)) std::array<unsigned char, BufferSize> buffer_{};
};
} // namespace retinify