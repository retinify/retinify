// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-eula

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace retinify
{
cudaError_t cudaLRConsistencyCheck(const float *leftDisparity, std::size_t leftDisparityStride,   //
                                   const float *rightDisparity, std::size_t rightDisparityStride, //
                                   float *outputDisparity, std::size_t outputDisparityStride,     //
                                   std::uint32_t disparityWidth, std::uint32_t disparityHeight,   //
                                   float maxRelativeDisparityError,                               //
                                   cudaStream_t stream);
} // namespace retinify
