// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-eula

#include "lrcheck.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#ifndef LRCC_BLOCK_W
#define LRCC_BLOCK_W 16
#endif
#ifndef LRCC_BLOCK_H
#define LRCC_BLOCK_H 16
#endif

namespace retinify
{
static inline std::uint32_t DivUpUint32(std::uint32_t a, std::uint32_t b)
{
    return (a + b - 1) / b;
}

__global__ void LRConsistencyCheckKernel(const float *__restrict__ leftDisparity, std::size_t leftDisparityStride,   //
                                         const float *__restrict__ rightDisparity, std::size_t rightDisparityStride, //
                                         float *__restrict__ outputDisparity, std::size_t outputDisparityStride,     //
                                         std::uint32_t disparityWidth, std::uint32_t disparityHeight,                //
                                         float maxRelativeDisparityError)
{
    const std::uint32_t x = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x) + static_cast<std::uint32_t>(threadIdx.x);
    const std::uint32_t y = static_cast<std::uint32_t>(blockIdx.y) * static_cast<std::uint32_t>(blockDim.y) + static_cast<std::uint32_t>(threadIdx.y);

    if (x >= disparityWidth || y >= disparityHeight)
    {
        return;
    }

    const std::size_t leftRowOffset = (static_cast<std::size_t>(y) * leftDisparityStride) / sizeof(float);
    const std::size_t rightRowOffset = (static_cast<std::size_t>(y) * rightDisparityStride) / sizeof(float);
    const std::size_t outputRowOffset = (static_cast<std::size_t>(y) * outputDisparityStride) / sizeof(float);

    const float *leftRow = leftDisparity + leftRowOffset;
    const float *rightRow = rightDisparity + rightRowOffset;
    float *outputRow = outputDisparity + outputRowOffset;

    float outputVal = 0.0f;
    const float ld = leftRow[x];
    if (ld > 0.0f && isfinite(ld))
    {
        const std::uint32_t roundedLd = static_cast<std::uint32_t>(ld + 0.5f);
        if (roundedLd <= x)
        {
            const std::uint32_t rx = x - roundedLd;
            if (rx < disparityWidth)
            {
                const float rd = rightRow[rx];
                if (isfinite(rd))
                {
                    const float absDiff = fabsf(ld - rd);
                    const float avgd = 0.5f * (fabsf(ld) + fabsf(rd));
                    const float relativeDiff = absDiff / avgd;
                    if (relativeDiff <= maxRelativeDisparityError)
                    {
                        outputVal = ld;
                    }
                }
            }
        }
    }

    outputRow[x] = outputVal;
}

cudaError_t cudaLRConsistencyCheck(const float *leftDisparity, std::size_t leftDisparityStride,   //
                                   const float *rightDisparity, std::size_t rightDisparityStride, //
                                   float *outputDisparity, std::size_t outputDisparityStride,     //
                                   std::uint32_t disparityWidth, std::uint32_t disparityHeight,   //
                                   float maxRelativeDisparityError,                               //
                                   cudaStream_t stream)
{
    if (leftDisparity == nullptr || rightDisparity == nullptr || outputDisparity == nullptr)
    {
        std::printf("Input pointer is null.\n");
        return cudaErrorInvalidValue;
    }

    if (disparityWidth == 0 || disparityHeight == 0)
    {
        std::printf("Input size must be positive.\n");
        return cudaErrorInvalidValue;
    }

    if ((leftDisparityStride % sizeof(float)) != 0 || (rightDisparityStride % sizeof(float)) != 0 || (outputDisparityStride % sizeof(float)) != 0)
    {
        std::printf("Stride must be a multiple of sizeof(float).\n");
        return cudaErrorInvalidValue;
    }

    if (maxRelativeDisparityError <= 0.0f || maxRelativeDisparityError >= 1.0f)
    {
        std::printf("maxRelativeDisparityError should be in the range (0.0, 1.0).\n");
        return cudaErrorInvalidValue;
    }

    dim3 block(LRCC_BLOCK_W, LRCC_BLOCK_H, 1);
    const std::uint32_t gridX = DivUpUint32(disparityWidth, static_cast<std::uint32_t>(block.x));
    const std::uint32_t gridY = DivUpUint32(disparityHeight, static_cast<std::uint32_t>(block.y));
    dim3 grid(gridX, gridY, 1);

    LRConsistencyCheckKernel<<<grid, block, 0, stream>>>(leftDisparity, leftDisparityStride,     //
                                                         rightDisparity, rightDisparityStride,   //
                                                         outputDisparity, outputDisparityStride, //
                                                         disparityWidth, disparityHeight,        //
                                                         maxRelativeDisparityError);

    return cudaGetLastError();
}
} // namespace retinify
