// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-eula

#pragma once

#include "retinify/geometry.hpp"
#include "retinify/status.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace retinify
{
/// @brief
/// The pixel format options for input images.
enum class PixelFormat : std::uint8_t
{
    /// @brief
    /// 8-bit 1ch, Grayscale format.
    GRAY8,
    /// @brief
    /// 8-bit 3ch, RGB format.
    RGB8,
};

/// @brief
/// The depth mode options for stereo matching pipeline.
enum class DepthMode : std::uint8_t
{
    /// @brief
    /// Fastest, with lowest accuracy.
    FAST,
    /// @brief
    /// Balanced, with moderate accuracy and speed.
    BALANCED,
    /// @brief
    /// Most accurate, with slowest performance.
    ACCURATE,
};

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
    /// @param imageWidth
    /// Width of the input images (in pixels).
    /// @param imageHeight
    /// Height of the input images (in pixels).
    /// @param pixelFormat
    /// The pixel format of the input images.
    /// @param depthMode
    /// The depth mode option for the stereo matching.
    /// @param calibrationParameters
    /// The stereo camera calibration parameters.
    /// @return
    /// A Status object indicating whether the initialization was successful.
    [[nodiscard]] auto Initialize(std::uint32_t imageWidth, std::uint32_t imageHeight,                                    //
                                  PixelFormat pixelFormat = PixelFormat::RGB8, DepthMode depthMode = DepthMode::ACCURATE, //
                                  const CalibrationParameters &calibrationParameters = CalibrationParameters{}) noexcept -> Status;

    /// @brief
    /// Executes the stereo matching pipeline using the given left and right image data.
    /// @param leftImageData
    /// Pointer to the left image data.
    /// @param leftImageStride
    /// Stride (in bytes) of a row in the left image.
    /// @param rightImageData
    /// Pointer to the right image data.
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
    /// Retrieves the rectified left image.
    /// @param leftImageData
    /// Pointer to the output buffer for left image data (8-bit unsigned char).
    /// @param leftImageStride
    /// Stride (in bytes) of a row in the output left image data.
    /// @return
    /// A Status object indicating whether the operation was successful.
    /// @note
    /// This function must be called after Run().
    [[nodiscard]] auto RetrieveRectifiedLeftImage(std::uint8_t *leftImageData, std::size_t leftImageStride) noexcept -> Status;

    /// @brief
    /// Retrieves the rectified right image.
    /// @param rightImageData
    /// Pointer to the output buffer for right image data (8-bit unsigned char).
    /// @param rightImageStride
    /// Stride (in bytes) of a row in the output right image data.
    /// @return
    /// A Status object indicating whether the operation was successful.
    /// @note
    /// This function must be called after Run().
    [[nodiscard]] auto RetrieveRectifiedRightImage(std::uint8_t *rightImageData, std::size_t rightImageStride) noexcept -> Status;

    /// @brief
    /// Retrieves the rectified left and right images.
    /// @param leftImageData
    /// Pointer to the output buffer for left image data (8-bit unsigned char).
    /// @param leftImageStride
    /// Stride (in bytes) of a row in the output left image data.
    /// @param rightImageData
    /// Pointer to the output buffer for right image data (8-bit unsigned char).
    /// @param rightImageStride
    /// Stride (in bytes) of a row in the output right image data.
    /// @return
    /// A Status object indicating whether the operation was successful.
    /// @note
    /// This function must be called after Run().
    [[nodiscard]] auto RetrieveRectifiedImages(std::uint8_t *leftImageData, std::size_t leftImageStride, //
                                               std::uint8_t *rightImageData, std::size_t rightImageStride) noexcept -> Status;

    /// @brief
    /// Retrieves the computed disparity map.
    /// @param disparityData
    /// Pointer to the output buffer for disparity data (32-bit float).
    /// @param disparityStride
    /// Stride (in bytes) of a row in the output disparity data.
    /// @return
    /// A Status object indicating whether the operation was successful.
    /// @note
    /// This function must be called after Run().
    [[nodiscard]] auto RetrieveDisparity(float *disparityData, std::size_t disparityStride) noexcept -> Status;

    /// @brief
    /// Reprojects the computed disparity map to a 3D point cloud.
    /// @param pointCloudData
    /// Pointer to the output buffer for point cloud data (32-bit float, 3 channels).
    /// @param pointCloudStride
    /// Stride (in bytes) of a row in the output point cloud buffer.
    /// @return
    /// A Status object indicating whether the operation was successful.
    /// @note
    /// This function must be called after Run().
    [[nodiscard]] auto RetrievePointCloud(float *pointCloudData, std::size_t pointCloudStride) noexcept -> Status;

  private:
    class Impl;
    auto impl() noexcept -> Impl *;
    [[nodiscard]] auto impl() const noexcept -> const Impl *;
    static constexpr std::size_t BufferSize{2048};
    alignas(alignof(std::max_align_t)) std::array<unsigned char, BufferSize> buffer_{};
};
} // namespace retinify
