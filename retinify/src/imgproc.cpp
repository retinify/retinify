// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "imgproc.hpp"

#include "retinify/logging.hpp"

#ifdef BUILD_WITH_TENSORRT
#include "cuda/lrcheck.h"
#include <npp.h>
#else
#endif

namespace retinify
{
auto ResizeImage8UC3(const Mat &src, Mat &dst) noexcept -> Status
{
    if (src.Empty() || dst.Empty())
    {
        LogError("Source or destination is empty.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (src.Channels() != 3 || dst.Channels() != 3)
    {
        LogError("Source and destination must have 3 channels.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    NppStatus status = nppiResize_8u_C3R(static_cast<const Npp8u *>(src.Data()), static_cast<int>(src.Stride()), //
                                         {static_cast<int>(src.Cols()), static_cast<int>(src.Rows())},           //
                                         {0, 0, static_cast<int>(src.Cols()), static_cast<int>(src.Rows())},     //
                                         static_cast<Npp8u *>(dst.Data()), static_cast<int>(dst.Stride()),       //
                                         {static_cast<int>(dst.Cols()), static_cast<int>(dst.Rows())},           //
                                         {0, 0, static_cast<int>(dst.Cols()), static_cast<int>(dst.Rows())},     //
                                         NPPI_INTER_LINEAR);

    if (status != NPP_SUCCESS)
    {
        LogError("nppiResize_8u_C3R failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    };

    return Status{};
#else
    (void)src;
    (void)dst;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto ResizeDisparity32FC1(const Mat &src, Mat &dst) noexcept -> Status
{
    if (src.Empty() || dst.Empty())
    {
        LogError("Source or destination is empty.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (src.Channels() != 1 || dst.Channels() != 1)
    {
        LogError("Source and destination must have 1 channel.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    NppStatus status = nppiResize_32f_C1R(static_cast<const Npp32f *>(src.Data()), static_cast<int>(src.Stride()), //
                                          {static_cast<int>(src.Cols()), static_cast<int>(src.Rows())},            //
                                          {0, 0, static_cast<int>(src.Cols()), static_cast<int>(src.Rows())},      //
                                          static_cast<Npp32f *>(dst.Data()), static_cast<int>(dst.Stride()),       //
                                          {static_cast<int>(dst.Cols()), static_cast<int>(dst.Rows())},            //
                                          {0, 0, static_cast<int>(dst.Cols()), static_cast<int>(dst.Rows())},      //
                                          NPPI_INTER_NN);

    if (status != NPP_SUCCESS)
    {
        LogError("nppiResize_32f_C1R failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    float value_scale = static_cast<float>(dst.Cols()) / static_cast<float>(src.Cols());
    status = nppiMulC_32f_C1IR(static_cast<Npp32f>(value_scale), static_cast<Npp32f *>(dst.Data()), static_cast<int>(dst.Stride()), {static_cast<int>(dst.Cols()), static_cast<int>(dst.Rows())});

    if (status != NPP_SUCCESS)
    {
        LogError("nppiMulC_32f_C1IR failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    return Status{};
#else
    (void)src;
    (void)dst;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto HorizontalFlip8UC3(const Mat &src, Mat &dst) noexcept -> Status
{
    if (src.Empty() || dst.Empty())
    {
        LogError("Source or destination is empty.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (src.Channels() != 3 || dst.Channels() != 3)
    {
        LogError("Source and destination must have 3 channels.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    NppStatus status = nppiMirror_8u_C3R(static_cast<const Npp8u *>(src.Data()), static_cast<int>(src.Stride()), //
                                         static_cast<Npp8u *>(dst.Data()), static_cast<int>(dst.Stride()),       //
                                         {static_cast<int>(src.Cols()), static_cast<int>(src.Rows())},           //
                                         NPP_VERTICAL_AXIS);

    if (status != NPP_SUCCESS)
    {
        LogError("nppiMirror_8u_C3R failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    };

    return Status{};
#else
    (void)src;
    (void)dst;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto HorizontalFlip32FC1(const Mat &src, Mat &dst) noexcept -> Status
{
    if (src.Empty() || dst.Empty())
    {
        LogError("Source or destination is empty.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (src.Channels() != 1 || dst.Channels() != 1)
    {
        LogError("Source and destination must have 1 channel.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    NppStatus status = nppiMirror_32f_C1R(static_cast<const Npp32f *>(src.Data()), static_cast<int>(src.Stride()), //
                                          static_cast<Npp32f *>(dst.Data()), static_cast<int>(dst.Stride()),       //
                                          {static_cast<int>(src.Cols()), static_cast<int>(src.Rows())},            //
                                          NPP_VERTICAL_AXIS);

    if (status != NPP_SUCCESS)
    {
        LogError("nppiMirror_32f_C1R failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    };

    return Status{};
#else
    (void)src;
    (void)dst;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto Convert8UC3To8UC1(const Mat &src, Mat &dst) noexcept -> Status
{
    if (src.Empty() || dst.Empty())
    {
        LogError("Source or destination is empty.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (src.Channels() != 3 || dst.Channels() != 1)
    {
        LogError("Source must have 3 channels and destination must have 1 channel.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    NppStatus status = nppiRGBToGray_8u_C3C1R(static_cast<const Npp8u *>(src.Data()), static_cast<int>(src.Stride()), //
                                              static_cast<Npp8u *>(dst.Data()), static_cast<int>(dst.Stride()),       //
                                              {static_cast<int>(src.Cols()), static_cast<int>(src.Rows())});

    if (status != NPP_SUCCESS)
    {
        LogError("nppiRGBToGray_8u_C3C1R failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    };

    return Status{};
#else
    (void)src;
    (void)dst;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto Convert8UC1To32FC1(const Mat &src, Mat &dst) noexcept -> Status
{
    if (src.Empty() || dst.Empty())
    {
        LogError("Source or destination is empty.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (src.Channels() != 1 || dst.Channels() != 1)
    {
        LogError("Source and destination must have 1 channel.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    NppStatus status = nppiConvert_8u32f_C1R(static_cast<const Npp8u *>(src.Data()), static_cast<int>(src.Stride()), //
                                             static_cast<Npp32f *>(dst.Data()), static_cast<int>(dst.Stride()),      //
                                             {static_cast<int>(src.Cols()), static_cast<int>(src.Rows())});

    if (status != NPP_SUCCESS)
    {
        LogError("nppiConvert_8u32f_C1R failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    };

    return Status{};
#else
    (void)src;
    (void)dst;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto Convert8UC3To32FC3(const Mat &src, Mat &dst) noexcept -> Status
{
    if (src.Empty() || dst.Empty())
    {
        LogError("Source or destination is empty.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (src.Channels() != 3 || dst.Channels() != 3)
    {
        LogError("Source and destination must have 3 channels.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    NppStatus status = nppiConvert_8u32f_C3R(static_cast<const Npp8u *>(src.Data()), static_cast<int>(src.Stride()), //
                                             static_cast<Npp32f *>(dst.Data()), static_cast<int>(dst.Stride()),      //
                                             {static_cast<int>(src.Cols()), static_cast<int>(src.Rows())});

    if (status != NPP_SUCCESS)
    {
        LogError("nppiConvert_8u32f_C3R failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    };

    return Status{};
#else
    (void)src;
    (void)dst;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto LRConsistencyCheck32FC1(const Mat &left, const Mat &right, Mat &output, float relativeError) noexcept -> Status
{
    if (left.Empty() || right.Empty() || output.Empty())
    {
        LogError("One of the input or output is empty.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (left.Channels() != 1 || right.Channels() != 1 || output.Channels() != 1)
    {
        LogError("All of the input and output must have 1 channel.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    cudaError_t error = cudaLRConsistencyCheck(static_cast<const float *>(left.Data()), left.Stride(),       //
                                               static_cast<const float *>(right.Data()), right.Stride(),     //
                                               static_cast<float *>(output.Data()), output.Stride(),         //
                                               static_cast<int>(left.Cols()), static_cast<int>(left.Rows()), //
                                               relativeError);

    if (error != cudaSuccess)
    {
        LogError("cudaLRConsistencyCheck failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    return Status{};
#else
    (void)left;
    (void)right;
    (void)output;
    (void)relativeError;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}
} // namespace retinify