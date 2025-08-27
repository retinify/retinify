// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifdef BUILD_WITH_TENSORRT
#include <npp.h>
#else
#endif

#include "mat.hpp"

#include <retinify/log.hpp>

namespace retinify
{
[[nodiscard]] auto MatResizeLinear8UC3(const Mat &src, Mat &dst) noexcept -> Status
{
    if (src.Channels() != 3 || dst.Channels() != 3)
    {
        LogError("Source and destination must have 3 channels.");
        return Status{StatusCategory::USER, StatusCode::INVALID_ARGUMENT};
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
        LogError("MatResizeLinear8UC3 failed");
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

[[nodiscard]] auto MatResizeNearest32FC1(const Mat &src, Mat &dst) noexcept -> Status
{
    if (src.Channels() != 1 || dst.Channels() != 1)
    {
        LogError("Source and destination must have 1 channel.");
        return Status{StatusCategory::USER, StatusCode::INVALID_ARGUMENT};
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

[[nodiscard]] auto Mat8UC3To8UC1(const Mat &src, Mat &dst) noexcept -> Status
{
    if (src.Channels() != 3 || dst.Channels() != 1)
    {
        LogError("Source must have 3 channels and destination must have 1 channel.");
        return Status{StatusCategory::USER, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    NppStatus status = nppiRGBToGray_8u_C3C1R(static_cast<const Npp8u *>(src.Data()), static_cast<int>(src.Stride()), //
                                              static_cast<Npp8u *>(dst.Data()), static_cast<int>(dst.Stride()),       //
                                              {static_cast<int>(src.Cols()), static_cast<int>(src.Rows())});

    if (status != NPP_SUCCESS)
    {
        LogError("Mat8UC3To8UC1 failed");
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

[[nodiscard]] auto MatConvert8UC1To32FC1(const Mat &src, Mat &dst) noexcept -> Status
{
    if (src.Channels() != 1 || dst.Channels() != 1)
    {
        LogError("Source and destination must have 1 channel.");
        return Status{StatusCategory::USER, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    NppStatus status = nppiConvert_8u32f_C1R(static_cast<const Npp8u *>(src.Data()), static_cast<int>(src.Stride()), //
                                             static_cast<Npp32f *>(dst.Data()), static_cast<int>(dst.Stride()),      //
                                             {static_cast<int>(src.Cols()), static_cast<int>(src.Rows())});

    if (status != NPP_SUCCESS)
    {
        LogError("MatCast8UTC1To32FC1 failed");
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
} // namespace retinify