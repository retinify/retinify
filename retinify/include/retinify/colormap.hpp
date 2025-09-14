// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/attributes.hpp"
#include "retinify/status.hpp"

#include <cstddef>
#include <cstdint>

namespace retinify
{
/// @brief
/// Applies the colormap to the disparity map.
/// @param src
/// Input disparity map (32-bit float).
/// @param srcStride
/// Stride of the input disparity map in bytes.
/// @param dst
/// Output colored disparity map (8-bit 3-channel RGB).
/// @param dstStride
/// Stride of the output colored disparity map in bytes.
/// @param height
/// Height of the input and output disparity maps.
/// @param width
/// Width of the input and output disparity maps.
/// @param maxDisp
/// Maximum disparity value for normalization.
/// @return
/// Status object indicating success or failure.
RETINIFY_API auto ColorizeDisparity(const float *src, size_t srcStride, //
                                    uint8_t *dst, size_t dstStride,     //
                                    int height, int width, float maxDisp) -> Status;
} // namespace retinify