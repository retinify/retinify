// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-eula

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace retinify
{
cudaError_t cudaDisparityToDepth(const float *disparity, std::size_t disparityStride, //
                                 float *depth, std::size_t depthStride,               //
                                 std::uint32_t width, std::uint32_t height,           //
                                 const float *reprojectionQ,                          //
                                 cudaStream_t stream);
} // namespace retinify
