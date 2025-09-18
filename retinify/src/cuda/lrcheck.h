// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace retinify
{
cudaError_t cudaLRConsistencyCheck(const float *leftDisparity, std::size_t leftDisparityStride,   //
                                   const float *rightDisparity, std::size_t rightDisparityStride, //
                                   float *outputDisparity, std::size_t outputDisparityStride,     //
                                   int disparityWidth, int disparityHeight,                       //
                                   float maxRelativeDisparityError,                               //
                                   cudaStream_t stream);
} // namespace retinify