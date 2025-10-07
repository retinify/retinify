// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-EULA

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace retinify
{
cudaError_t cudaDisparityOcclusionFilter(const float *leftDisparity, std::size_t leftDisparityStride, //
                                         float *outputDisparity, std::size_t outputDisparityStride,   //
                                         std::uint32_t disparityWidth, std::uint32_t disparityHeight, //
                                         cudaStream_t stream);
} // namespace retinify