// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-EULA

#include "occlusion.h"

#include <cfloat>
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <limits>
#include <thrust/functional.h>

#ifndef OCCL_BLOCK_W
#define OCCL_BLOCK_W 16
#endif
#ifndef OCCL_BLOCK_H
#define OCCL_BLOCK_H 16
#endif

namespace retinify
{
static inline std::uint32_t DivUpUint32(std::uint32_t a, std::uint32_t b)
{
    return (a + b - 1) / b;
}

__global__ void DisparityOcclusionFilterKernel(const float *__restrict__ leftDisparity, std::size_t leftDisparityStride, //
                                               float *__restrict__ outputDisparity, std::size_t outputDisparityStride,   //
                                               std::uint32_t disparityWidth, std::uint32_t disparityHeight)
{
    extern __shared__ float sharedCarry[];

    const std::uint32_t localX = static_cast<std::uint32_t>(threadIdx.x);
    const std::uint32_t localY = static_cast<std::uint32_t>(threadIdx.y);
    const std::uint32_t y = static_cast<std::uint32_t>(blockIdx.y) * static_cast<std::uint32_t>(blockDim.y) + localY;

    const bool rowInBounds = y < disparityHeight;
    const int blockWidth = static_cast<int>(blockDim.x);

    cooperative_groups::thread_block blockGroup = cooperative_groups::this_thread_block();
    cooperative_groups::thread_block_tile<OCCL_BLOCK_W> rowGroup = cooperative_groups::tiled_partition<OCCL_BLOCK_W>(blockGroup);

    if (localX == 0)
    {
        sharedCarry[localY] = FLT_MAX;
    }
    blockGroup.sync();

    const int width = static_cast<int>(disparityWidth);

    const std::size_t leftRowOffset = rowInBounds ? (static_cast<std::size_t>(y) * leftDisparityStride) / sizeof(float) : 0U;
    const std::size_t outputRowOffset = rowInBounds ? (static_cast<std::size_t>(y) * outputDisparityStride) / sizeof(float) : 0U;

    const float *leftRow = rowInBounds ? (leftDisparity + leftRowOffset) : nullptr;
    float *outputRow = rowInBounds ? (outputDisparity + outputRowOffset) : nullptr;

    thrust::minimum<float> min_op;

    for (int remaining = width; remaining > 0; remaining -= blockWidth)
    {
        const int tileCount = remaining < blockWidth ? remaining : blockWidth;
        const bool inTile = rowInBounds && (localX < static_cast<std::uint32_t>(tileCount));
        const int x = remaining - 1 - static_cast<int>(localX);

        float disparity = 0.0f;
        float projectedRight = FLT_MAX;
        bool validDisparity = false;

        if (inTile)
        {
            disparity = leftRow[x];
            if (disparity > 0.0f && isfinite(disparity))
            {
                projectedRight = static_cast<float>(x) - disparity;
                if (isfinite(projectedRight) && (projectedRight >= 0.0f) && (projectedRight <= static_cast<float>(width - 1)))
                {
                    validDisparity = true;
                }
                else
                {
                    projectedRight = FLT_MAX;
                }
            }
            else
            {
                projectedRight = FLT_MAX;
            }
        }

        const float carry = sharedCarry[localY];
        const float tileValue = inTile ? projectedRight : FLT_MAX;

        float prefixMin = cooperative_groups::exclusive_scan(rowGroup, tileValue, min_op);
        if (rowGroup.thread_rank() == 0)
        {
            prefixMin = FLT_MAX;
        }
        const float minRight = fminf(carry, prefixMin);

        if (inTile)
        {
            outputRow[x] = (validDisparity && (projectedRight < minRight)) ? disparity : 0.0f;
        }

        const float tileMin = cooperative_groups::reduce(rowGroup, tileValue, min_op);
        if (rowGroup.thread_rank() == 0)
        {
            sharedCarry[localY] = fminf(carry, tileMin);
        }
        blockGroup.sync();
    }
}

cudaError_t cudaDisparityOcclusionFilter(const float *leftDisparity, std::size_t leftDisparityStride, //
                                         float *outputDisparity, std::size_t outputDisparityStride,   //
                                         std::uint32_t disparityWidth, std::uint32_t disparityHeight, //
                                         cudaStream_t stream)
{
    if (leftDisparity == nullptr || outputDisparity == nullptr)
    {
        std::printf("Input pointer is null.\n");
        return cudaErrorInvalidValue;
    }

    if (disparityWidth == 0 || disparityHeight == 0)
    {
        std::printf("Input size must be positive.\n");
        return cudaErrorInvalidValue;
    }

    if ((leftDisparityStride % sizeof(float)) != 0 || (outputDisparityStride % sizeof(float)) != 0)
    {
        std::printf("Stride must be a multiple of sizeof(float).\n");
        return cudaErrorInvalidValue;
    }

    if (disparityWidth > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
    {
        std::printf("Disparity width is too large for internal indexing.\n");
        return cudaErrorInvalidValue;
    }

    dim3 block(OCCL_BLOCK_W, OCCL_BLOCK_H, 1);
    const std::uint32_t gridY = DivUpUint32(disparityHeight, static_cast<std::uint32_t>(block.y));
    dim3 grid(1, gridY, 1);

    const std::size_t sharedBytes = static_cast<std::size_t>(block.y) * sizeof(float);

    DisparityOcclusionFilterKernel<<<grid, block, sharedBytes, stream>>>(leftDisparity, leftDisparityStride,     //
                                                                         outputDisparity, outputDisparityStride, //
                                                                         disparityWidth, disparityHeight);

    return cudaGetLastError();
}
} // namespace retinify
