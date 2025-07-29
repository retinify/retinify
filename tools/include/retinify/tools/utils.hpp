// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/define.hpp"

#include <opencv2/core.hpp>

namespace retinify::tools
{
/// @brief
/// Colorizes a disparity map.
/// @param disparity
/// The disparity map to colorize as a `cv::Mat`.
/// @param maxDisparity
/// The maximum disparity value used for normalization.
/// @return
/// A colorized disparity map as a `cv::Mat`.
RETINIFY_API [[nodiscard]] cv::Mat ColorizeDisparity(const cv::Mat &disparity, int maxDisparity);
} // namespace retinify::tools