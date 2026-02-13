// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "attributes.hpp"
#include "status.hpp"

#include <cstddef>
#include <cstdint>

namespace retinify
{
/// @brief
/// Resize an 8-bit image using bilinear interpolation
/// @param src
/// Input image data pointer
/// @param srcStride
/// Stride of a row in the source image (in bytes)
/// @param dst
/// Output image data pointer
/// @param dstStride
/// Stride of a row in the destination image (in bytes)
/// @param srcWidth
/// Source image width (in pixels)
/// @param srcHeight
/// Source image height (in pixels)
/// @param dstWidth
/// Destination image width (in pixels)
/// @param dstHeight
/// Destination image height (in pixels)
/// @param channels
/// Number of channels (1 or 3)
/// @return
/// A Status object that indicates whether the operation was successful
RETINIFY_API auto Resize(const std::uint8_t *src, std::size_t srcStride, std::uint8_t *dst, std::size_t dstStride, std::size_t srcWidth, std::size_t srcHeight, std::size_t dstWidth, std::size_t dstHeight, std::size_t channels) noexcept -> Status;

/// @brief
/// Remap an 8-bit image using the provided x/y coordinate maps
/// @param src
/// Input image data pointer
/// @param srcStride
/// Stride of a row in the source image (in bytes)
/// @param dst
/// Output image data pointer
/// @param dstStride
/// Stride of a row in the destination image (in bytes)
/// @param mapX
/// Map for x-coordinates
/// @param mapXStride
/// Stride of a row in mapX (in bytes)
/// @param mapY
/// Map for y-coordinates
/// @param mapYStride
/// Stride of a row in mapY (in bytes)
/// @param imageWidth
/// Image width (in pixels)
/// @param imageHeight
/// Image height (in pixels)
/// @param channels
/// Number of channels (1 or 3)
/// @return
/// A Status object that indicates whether the operation was successful
RETINIFY_API auto Remap(const std::uint8_t *src, std::size_t srcStride, std::uint8_t *dst, std::size_t dstStride, const float *mapX, std::size_t mapXStride, const float *mapY, std::size_t mapYStride, std::size_t imageWidth, std::size_t imageHeight, std::size_t channels) noexcept -> Status;
} // namespace retinify
