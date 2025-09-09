// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mat.hpp"

namespace retinify
{
/// @brief
/// Resize an 8-bit, 3-channel image using bilinear interpolation.
/// @param src
/// Input image (8-bit, 3-channel).
/// @param dst
/// Output image (8-bit, 3-channel).
/// @return
/// Status code indicating success or failure.
[[nodiscard]] auto ResizeImage8UC3(const Mat &src, Mat &dst) noexcept -> Status;

/// @brief
/// Resize a 32-bit floating-point, 1-channel disparity map using nearest-neighbor interpolation.
/// @param src
/// Input disparity map (32-bit floating-point, 1-channel).
/// @param dst
/// Output disparity map (32-bit floating-point, 1-channel).
/// @return
/// Status code indicating success or failure.
[[nodiscard]] auto ResizeDisparity32FC1(const Mat &src, Mat &dst) noexcept -> Status;

/// @brief
/// Horizontally flip an 8-bit, 1-channel image.
/// @param src
/// Input image (8-bit, 1-channel).
/// @param dst
/// Output image (8-bit, 1-channel).
/// @return
/// Status code indicating success or failure.
[[nodiscard]] auto HorizontalFlip8UC3(const Mat &src, Mat &dst) noexcept -> Status;

/// @brief
/// Horizontally flip a 32-bit floating-point, 1-channel image.
/// @param src
/// Input image (32-bit floating-point, 1-channel).
/// @param dst
/// Output image (32-bit floating-point, 1-channel).
/// @return
/// Status code indicating success or failure.
[[nodiscard]] auto HorizontalFlip32FC1(const Mat &src, Mat &dst) noexcept -> Status;

/// @brief
/// Convert an 8-bit, 3-channel image to an 8-bit, 1-channel grayscale image.
/// @param src
/// Input image (8-bit, 3-channel).
/// @param dst
/// Output grayscale image (8-bit, 1-channel).
/// @return
/// Status code indicating success or failure.
[[nodiscard]] auto Convert8UC3To8UC1(const Mat &src, Mat &dst) noexcept -> Status;

/// @brief
/// Convert an 8-bit, 1-channel grayscale image to a 32-bit floating-point, 1-channel image.
/// @param src
/// Input grayscale image (8-bit, 1-channel).
/// @param dst
/// Output image (32-bit floating-point, 1-channel).
/// @return
/// Status code indicating success or failure.
[[nodiscard]] auto Convert8UC1To32FC1(const Mat &src, Mat &dst) noexcept -> Status;

/// @brief
/// Convert an 8-bit, 3-channel image to a 32-bit floating-point, 3-channel image.
/// @param src
/// Input image (8-bit, 3-channel).
/// @param dst
/// Output image (32-bit floating-point, 3-channel).
/// @return
/// Status code indicating success or failure.
[[nodiscard]] auto Convert8UC3To32FC3(const Mat &src, Mat &dst) noexcept -> Status;

/// @brief
/// Perform left-right consistency check on two 32-bit floating-point, 1-channel disparity maps.
/// @param left
/// Input left disparity map (32-bit floating-point, 1-channel).
/// @param right
/// Input right disparity map (32-bit floating-point, 1-channel).
/// @param output
/// Output disparity map after consistency check (32-bit floating-point, 1-channel).
/// @param maxDifference
/// Maximum allowable difference between left and right disparities for consistency.
/// @return
[[nodiscard]] auto LRConsistencyCheck32FC1(const Mat &left, const Mat &right, Mat &output, float maxDifference) noexcept -> Status;
} // namespace retinify