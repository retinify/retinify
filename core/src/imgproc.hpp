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
[[nodiscard]] auto Convert8UC3To8UC1(const Mat &src, Mat &dst) noexcept -> Status;
[[nodiscard]] auto Convert8UC1To32FC1(const Mat &src, Mat &dst) noexcept -> Status;
} // namespace retinify