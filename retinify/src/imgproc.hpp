// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-EULA

#include "mat.hpp"
#include "stream.hpp"

namespace retinify
{
/// @brief
/// Resize an 8-bit, 1- or 3-channel image using bilinear interpolation.
/// @param src
/// Input image (8-bit, 1- or 3-channel).
/// @param dst
/// Output image (8-bit, 1- or 3-channel).
/// @param stream
/// Execution stream.
/// @return
/// Status code.
[[nodiscard]] auto ResizeImage8U(const Mat &src, Mat &dst, Stream &stream) noexcept -> Status;

/// @brief
/// Resize a 32-bit floating-point, 1-channel disparity map using nearest-neighbor interpolation.
/// @param src
/// Input disparity map (32-bit floating-point, 1-channel).
/// @param dst
/// Output disparity map (32-bit floating-point, 1-channel).
/// @param stream
/// Execution stream.
/// @return
/// Status code.
[[nodiscard]] auto ResizeDisparity32FC1(const Mat &src, Mat &dst, Stream &stream) noexcept -> Status;

/// @brief
/// Convert an 8-bit image to an 8-bit, 1-channel grayscale image.
/// @param src
/// Input image (8-bit, 1- or 3-channel).
/// @param dst
/// Output grayscale image (8-bit, 1-channel). Must have the same size as the input.
/// @param stream
/// Execution stream.
/// @return
/// Status code.
[[nodiscard]] auto ConvertImage8UToC1(const Mat &src, Mat &dst, Stream &stream) noexcept -> Status;

/// @brief
/// Convert an 8-bit, 1-channel grayscale image to a 32-bit floating-point, 1-channel image.
/// @param src
/// Input grayscale image (8-bit, 1-channel).
/// @param dst
/// Output image (32-bit floating-point, 1-channel).
/// @param stream
/// Execution stream.
/// @return
/// Status code.
[[nodiscard]] auto Convert8UC1To32FC1(const Mat &src, Mat &dst, Stream &stream) noexcept -> Status;

/// @brief
/// Convert an 8-bit, 3-channel image to a 32-bit floating-point, 3-channel image.
/// @param src
/// Input image (8-bit, 3-channel).
/// @param dst
/// Output image (32-bit floating-point, 3-channel).
/// @param stream
/// Execution stream.
/// @return
/// Status code.
[[nodiscard]] auto Convert8UC3To32FC3(const Mat &src, Mat &dst, Stream &stream) noexcept -> Status;

/// @brief
/// Remove occluded pixels from a 32-bit floating-point, 1-channel left disparity map.
/// @param src
/// Input left disparity map (32-bit floating-point, 1-channel).
/// @param dst
/// Output disparity map with occlusions removed (32-bit floating-point, 1-channel).
/// @param stream
/// Execution stream.
/// @return
/// Status code.
[[nodiscard]] auto DisparityOcclusionFilter32FC1(const Mat &src, Mat &dst, Stream &stream) noexcept -> Status;

/// @brief
/// Remap an 8-bit, 1-channel image using the provided x and y maps.
/// @param src
/// Input image (8-bit, 1- or 3-channel).
/// @param mapX
/// X map (32-bit floating-point, 1-channel).
/// @param mapY
/// Y map (32-bit floating-point, 1-channel).
/// @param dst
/// Output image (8-bit, 1- or 3-channel).
/// @param stream
/// Execution stream.
/// @return
/// Status code.
[[nodiscard]] auto RemapImage8U(const Mat &src, const Mat &mapX, const Mat &mapY, Mat &dst, Stream &stream) noexcept -> Status;

/// @brief
/// Horizontally flip an 8-bit, 1-channel image.
/// @param src
/// Input image (8-bit, 1-channel).
/// @param dst
/// Output image (8-bit, 1-channel).
/// @param stream
/// Execution stream.
/// @return
/// Status code.
[[nodiscard]] auto HorizontalFlip8UC3(const Mat &src, Mat &dst, Stream &stream) noexcept -> Status;

/// @brief
/// Horizontally flip a 32-bit floating-point, 1-channel image.
/// @param src
/// Input image (32-bit floating-point, 1-channel).
/// @param dst
/// Output image (32-bit floating-point, 1-channel).
/// @param stream
/// Execution stream.
/// @return
/// Status code.
[[nodiscard]] auto HorizontalFlip32FC1(const Mat &src, Mat &dst, Stream &stream) noexcept -> Status;

/// @brief
/// Perform left-right consistency check on two 32-bit floating-point, 1-channel disparity maps.
/// @param left
/// Input left disparity map (32-bit floating-point, 1-channel).
/// @param right
/// Input right disparity map (32-bit floating-point, 1-channel).
/// @param output
/// Output disparity map after consistency check (32-bit floating-point, 1-channel).
/// @param relativeError
/// Maximum allowed relative error for consistency check.
/// @param stream
/// Execution stream.
/// @return
/// Status code.
[[nodiscard]] auto LRConsistencyCheck32FC1(const Mat &left, const Mat &right, Mat &output, float relativeError, Stream &stream) noexcept -> Status;
} // namespace retinify
