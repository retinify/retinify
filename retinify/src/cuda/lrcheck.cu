// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "lrcheck.h"

#include <cmath>
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
static inline int DivUpInt(int a, int b)
{
    return (a + b - 1) / b;
}

__global__ void LRConsistencyCheckKernel(const float *__restrict__ leftDisparity, std::size_t leftDisparityStride,   //
                                         const float *__restrict__ rightDisparity, std::size_t rightDisparityStride, //
                                         float *__restrict__ outputDisparity, std::size_t outputDisparityStride,     //
                                         int disparityWidth, int disparityHeight,                                    //
                                         float maxDisparityDifference)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= disparityWidth || y >= disparityHeight)
    {
        return;
    }

    const char *leftRowBase = reinterpret_cast<const char *>(leftDisparity) + static_cast<std::size_t>(y) * leftDisparityStride;
    const char *rightRowBase = reinterpret_cast<const char *>(rightDisparity) + static_cast<std::size_t>(y) * rightDisparityStride;
    char *outputRowBase = reinterpret_cast<char *>(outputDisparity) + static_cast<std::size_t>(y) * outputDisparityStride;

    const float *leftRow = reinterpret_cast<const float *>(leftRowBase);
    const float *rightRow = reinterpret_cast<const float *>(rightRowBase);
    float *outputRow = reinterpret_cast<float *>(outputRowBase);

    float outputVal = 0.0f;
    const float ld = leftRow[x];
    if (ld > 0.0f && isfinite(ld))
    {
        const int rx = x - static_cast<int>(ld + 0.5f);
        if (rx >= 0 && rx < disparityWidth)
        {
            const float rd = rightRow[rx];
            if (isfinite(rd))
            {
                const float diff = fabsf(ld - rd);
                if (diff <= maxDisparityDifference)
                {
                    outputVal = ld;
                }
            }
        }
    }

    outputRow[x] = outputVal;
}

cudaError_t cudaLRConsistencyCheck(const float *leftDisparity, std::size_t leftDisparityStride,   //
                                   const float *rightDisparity, std::size_t rightDisparityStride, //
                                   float *outputDisparity, std::size_t outputDisparityStride,     //
                                   int disparityWidth, int disparityHeight,                       //
                                   float maxDisparityDifference,                                  //
                                   cudaStream_t stream)
{
    if (leftDisparity == nullptr || rightDisparity == nullptr || outputDisparity == nullptr)
    {
        std::printf("Input pointer is null.\n");
        return cudaErrorInvalidValue;
    }

    if (disparityWidth <= 0 || disparityHeight <= 0)
    {
        std::printf("Input size must be positive.\n");
        return cudaErrorInvalidValue;
    }

    if ((leftDisparityStride % sizeof(float)) != 0 || (rightDisparityStride % sizeof(float)) != 0 || (outputDisparityStride % sizeof(float)) != 0)
    {
        std::printf("Stride must be a multiple of sizeof(float).\n");
        return cudaErrorInvalidValue;
    }

    dim3 block(LRCC_BLOCK_W, LRCC_BLOCK_H, 1);
    dim3 grid(DivUpInt(disparityWidth, block.x), DivUpInt(disparityHeight, block.y), 1);

    LRConsistencyCheckKernel<<<grid, block, 0, stream>>>(leftDisparity, leftDisparityStride,     //
                                                         rightDisparity, rightDisparityStride,   //
                                                         outputDisparity, outputDisparityStride, //
                                                         disparityWidth, disparityHeight,        //
                                                         maxDisparityDifference);

    return cudaGetLastError();
}
} // namespace retinify