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
[[nodiscard]] auto GrayMatCast8UTo32F(const Mat &src, Mat &dst) noexcept -> Status
{
#ifdef BUILD_WITH_TENSORRT
    NppStatus status = nppiConvert_8u32f_C1R(static_cast<const Npp8u *>(src.Data()), static_cast<int>(src.Stride()), //
                                             static_cast<Npp32f *>(dst.Data()), static_cast<int>(dst.Stride()),      //
                                             {static_cast<int>(src.Cols()), static_cast<int>(src.Rows())});

    if (status != NPP_SUCCESS)
    {
        LogError("GrayMatCast8UTo32F failed");
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