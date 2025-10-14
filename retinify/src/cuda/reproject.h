// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-EULA

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace retinify
{
cudaError_t cudaReprojectTo3d(const float *disparity, std::size_t disparityStride, //
                              float *points3d, std::size_t points3dStride,         //
                              std::uint32_t width, std::uint32_t height,           //
                              const float *reprojectionQ,                          //
                              cudaStream_t stream);
} // namespace retinify
