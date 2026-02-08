// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cuda_common.cuh"
#include "cuda_occlusion.cuh"

#include <cfloat>
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <thrust/functional.h>

namespace retinify
{
__global__ void DisparityOcclusionFilterKernel(const float *__restrict__ leftDisparity, std::size_t leftDisparityStride, //
                                               float *__restrict__ outputDisparity, std::size_t outputDisparityStride,   //
                                               std::uint32_t disparityWidth, std::uint32_t disparityHeight)
{
    extern __shared__ float sharedCarry[];

    const std::uint32_t threadX = static_cast<std::uint32_t>(threadIdx.x);
    const std::uint32_t threadY = static_cast<std::uint32_t>(threadIdx.y);
    cooperative_groups::thread_block blockGroup = cooperative_groups::this_thread_block();
    cooperative_groups::thread_block_tile<kBlockW> rowGroup = cooperative_groups::tiled_partition<kBlockW>(blockGroup);

    const std::uint32_t y = static_cast<std::uint32_t>(blockIdx.y) * static_cast<std::uint32_t>(blockDim.y) + threadY;
    const bool rowInBounds = y < disparityHeight;
    const int tileWidth = static_cast<int>(blockDim.x);
    const int width = static_cast<int>(disparityWidth);
    const float widthMax = static_cast<float>(width - 1);
    const int lane = rowGroup.thread_rank();

    if (threadX == 0)
    {
        sharedCarry[threadY] = FLT_MAX;
    }
    blockGroup.sync();

    const std::size_t leftRowOffset = rowInBounds ? (static_cast<std::size_t>(y) * leftDisparityStride) / sizeof(float) : 0U;
    const std::size_t outputRowOffset = rowInBounds ? (static_cast<std::size_t>(y) * outputDisparityStride) / sizeof(float) : 0U;

    const float *leftRow = rowInBounds ? (leftDisparity + leftRowOffset) : nullptr;
    float *outputRow = rowInBounds ? (outputDisparity + outputRowOffset) : nullptr;

    const thrust::minimum<float> minOp;

    for (int remaining = width; remaining > 0; remaining -= tileWidth)
    {
        const int x = remaining - 1 - static_cast<int>(threadX);
        const bool inTile = rowInBounds && (x >= 0);

        float disparity = 0.0f;
        float projectedRight = FLT_MAX;
        bool validDisparity = false;

        if (inTile)
        {
            disparity = leftRow[x];
            if (disparity > 0.0f && isfinite(disparity))
            {
                const float right = static_cast<float>(x) - disparity;
                if (isfinite(right) && (right >= 0.0f) && (right <= widthMax))
                {
                    projectedRight = right;
                    validDisparity = true;
                }
            }
        }

        const float carry = sharedCarry[threadY];
        const float tileValue = projectedRight;

        float prefixMin = cooperative_groups::exclusive_scan(rowGroup, tileValue, minOp);
        if (lane == 0)
        {
            prefixMin = FLT_MAX;
        }
        const float minRight = fminf(carry, prefixMin);

        if (inTile)
        {
            outputRow[x] = (validDisparity && (projectedRight < minRight)) ? disparity : 0.0f;
        }

        const float tileMin = cooperative_groups::reduce(rowGroup, tileValue, minOp);
        if (lane == 0)
        {
            sharedCarry[threadY] = fminf(carry, tileMin);
        }
        blockGroup.sync();
    }
}

cudaError_t cudaDisparityOcclusionFilter(const float *leftDisparity, std::size_t leftDisparityStride, //
                                         float *outputDisparity, std::size_t outputDisparityStride,   //
                                         std::uint32_t disparityWidth, std::uint32_t disparityHeight, cudaStream_t stream)
{
    if (leftDisparity == nullptr || outputDisparity == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    if (disparityWidth == 0U || disparityHeight == 0U)
    {
        return cudaErrorInvalidValue;
    }

    const std::size_t requiredLeftDisparityStride = static_cast<std::size_t>(disparityWidth) * sizeof(float);
    const std::size_t requiredOutputDisparityStride = static_cast<std::size_t>(disparityWidth) * sizeof(float);

    if ((leftDisparityStride % sizeof(float)) != 0 || (outputDisparityStride % sizeof(float)) != 0 || leftDisparityStride < requiredLeftDisparityStride || outputDisparityStride < requiredOutputDisparityStride)
    {
        return cudaErrorInvalidValue;
    }

    dim3 block(kBlockW, kBlockH, 1);
    dim3 grid(1, DivUp(disparityHeight, static_cast<std::uint32_t>(block.y)), 1);

    const std::size_t sharedBytes = static_cast<std::size_t>(block.y) * sizeof(float);

    DisparityOcclusionFilterKernel<<<grid, block, sharedBytes, stream>>>(leftDisparity, leftDisparityStride, outputDisparity, outputDisparityStride, disparityWidth, disparityHeight);

    return cudaGetLastError();
}
} // namespace retinify
