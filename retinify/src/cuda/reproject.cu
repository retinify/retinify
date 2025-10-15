// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-eula

#include "reproject.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#ifndef REPROJECT_BLOCK_W
#define REPROJECT_BLOCK_W 16
#endif

#ifndef REPROJECT_BLOCK_H
#define REPROJECT_BLOCK_H 16
#endif

namespace retinify
{
namespace
{
struct Matrix4x4f
{
    float4 rows[4];
};

__device__ __forceinline__ float MultiplyRow(const Matrix4x4f &matrix, int row, float u, float v, float disparity)
{
    const float4 r = matrix.rows[row];
    return (r.x * u) + (r.y * v) + (r.z * disparity) + r.w;
}

__host__ __device__ __forceinline__ std::uint32_t DivUp(std::uint32_t value, std::uint32_t divisor)
{
    return (value + divisor - 1U) / divisor;
}
} // namespace

__global__ void ReprojectTo3dKernel(const float *__restrict__ disparity, std::size_t disparityStride, //
                                    float *__restrict__ points3d, std::size_t points3dStride,         //
                                    std::uint32_t width, std::uint32_t height,                        //
                                    Matrix4x4f reprojection)
{
    const std::uint32_t x = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x) + static_cast<std::uint32_t>(threadIdx.x);
    const std::uint32_t y = static_cast<std::uint32_t>(blockIdx.y) * static_cast<std::uint32_t>(blockDim.y) + static_cast<std::uint32_t>(threadIdx.y);

    if (x >= width || y >= height)
    {
        return;
    }

    const std::size_t disparityRowOffset = (static_cast<std::size_t>(y) * disparityStride) / sizeof(float);
    const std::size_t points3dRowOffset = (static_cast<std::size_t>(y) * points3dStride) / sizeof(float);

    const float *disparityRow = disparity + disparityRowOffset;
    float *pointRow = points3d + points3dRowOffset;

    float outX = 0.0f;
    float outY = 0.0f;
    float outZ = 0.0f;

    const float disparityValue = disparityRow[x];
    if (disparityValue > 0.0f && isfinite(disparityValue))
    {
        const float u = static_cast<float>(x);
        const float v = static_cast<float>(y);

        const float X = MultiplyRow(reprojection, 0, u, v, disparityValue);
        const float Y = MultiplyRow(reprojection, 1, u, v, disparityValue);
        const float Z = MultiplyRow(reprojection, 2, u, v, disparityValue);
        const float W = MultiplyRow(reprojection, 3, u, v, disparityValue);

        if (fabsf(W) > 1e-6f)
        {
            const float invW = 1.0f / W;
            outX = X * invW;
            outY = Y * invW;
            outZ = Z * invW;
        }
    }

    const std::size_t idx = static_cast<std::size_t>(x) * 3;
    pointRow[idx + 0] = outX;
    pointRow[idx + 1] = outY;
    pointRow[idx + 2] = outZ;
}

cudaError_t cudaReprojectTo3d(const float *disparity, std::size_t disparityStride, //
                              float *points3d, std::size_t points3dStride,         //
                              std::uint32_t width, std::uint32_t height,           //
                              const float *reprojectionQ,                          //
                              cudaStream_t stream)
{
    if (disparity == nullptr || points3d == nullptr || reprojectionQ == nullptr)
    {
        std::printf("Input pointer is null.\n");
        return cudaErrorInvalidValue;
    }

    if (width == 0U || height == 0U)
    {
        std::printf("Input size must be positive.\n");
        return cudaErrorInvalidValue;
    }

    if ((disparityStride % sizeof(float)) != 0U || (points3dStride % sizeof(float)) != 0U)
    {
        std::printf("Stride must be a multiple of sizeof(float).\n");
        return cudaErrorInvalidValue;
    }

    const std::size_t requiredDisparityStride = static_cast<std::size_t>(width) * sizeof(float);
    if (disparityStride < requiredDisparityStride)
    {
        std::printf("Disparity stride is too small.\n");
        return cudaErrorInvalidValue;
    }

    const std::size_t requiredPointsStride = static_cast<std::size_t>(width) * 3U * sizeof(float);
    if (points3dStride < requiredPointsStride)
    {
        std::printf("Point stride is too small.\n");
        return cudaErrorInvalidValue;
    }

    Matrix4x4f matrix{};
    matrix.rows[0] = make_float4(reprojectionQ[0], reprojectionQ[1], reprojectionQ[2], reprojectionQ[3]);
    matrix.rows[1] = make_float4(reprojectionQ[4], reprojectionQ[5], reprojectionQ[6], reprojectionQ[7]);
    matrix.rows[2] = make_float4(reprojectionQ[8], reprojectionQ[9], reprojectionQ[10], reprojectionQ[11]);
    matrix.rows[3] = make_float4(reprojectionQ[12], reprojectionQ[13], reprojectionQ[14], reprojectionQ[15]);

    dim3 block(REPROJECT_BLOCK_W, REPROJECT_BLOCK_H, 1);
    dim3 grid(DivUp(width, static_cast<std::uint32_t>(block.x)), DivUp(height, static_cast<std::uint32_t>(block.y)), 1);

    ReprojectTo3dKernel<<<grid, block, 0, stream>>>(disparity, disparityStride, //
                                                    points3d, points3dStride,   //
                                                    width, height,              //
                                                    matrix);

    return cudaGetLastError();
}
} // namespace retinify
