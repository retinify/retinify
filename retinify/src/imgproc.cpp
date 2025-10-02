// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "imgproc.hpp"

#include "retinify/logging.hpp"

#ifdef BUILD_WITH_TENSORRT
#include "cuda/lrcheck.h"
#include "cuda/occlusion.h"
#include <npp.h>
#else
#endif

namespace retinify
{
auto ResizeImage8U(const Mat &src, Mat &dst, Stream &stream) noexcept -> Status
{
    if (src.Empty() || dst.Empty())
    {
        LogError("Source or destination is empty.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (src.Channels() != dst.Channels())
    {
        LogError("Source and destination must have the same number of channels.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (src.Channels() != 1 && src.Channels() != 3)
    {
        LogError("Source and destination must have 1 or 3 channels.");
        return Status{};
    }

#ifdef BUILD_WITH_TENSORRT
    const auto srcData = static_cast<const Npp8u *>(src.Data());
    const auto dstData = static_cast<Npp8u *>(dst.Data());
    const auto srcStride = static_cast<int>(src.Stride());
    const auto dstStride = static_cast<int>(dst.Stride());
    const auto srcSize = NppiSize{static_cast<int>(src.Cols()), static_cast<int>(src.Rows())};
    const auto dstSize = NppiSize{static_cast<int>(dst.Cols()), static_cast<int>(dst.Rows())};
    const auto srcRoi = NppiRect{0, 0, srcSize.width, srcSize.height};
    const auto dstRoi = NppiRect{0, 0, dstSize.width, dstSize.height};

    NppStatus status{};

    if (src.Channels() == 1)
    {
        status = nppiResize_8u_C1R_Ctx(srcData, srcStride, //
                                       srcSize, srcRoi,    //
                                       dstData, dstStride, //
                                       dstSize, dstRoi,    //
                                       NPPI_INTER_LINEAR, stream.GetNppStreamContext());
    }
    else // src.Channels() == 3
    {
        status = nppiResize_8u_C3R_Ctx(srcData, srcStride, //
                                       srcSize, srcRoi,    //
                                       dstData, dstStride, //
                                       dstSize, dstRoi,    //
                                       NPPI_INTER_LINEAR, stream.GetNppStreamContext());
    }

    if (status != NPP_SUCCESS)
    {
        LogError(src.Channels() == 1 ? "nppiResize_8u_C1R failed" : "nppiResize_8u_C3R failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    };

    return Status{};
#else
    (void)src;
    (void)dst;
    (void)stream;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto ResizeDisparity32FC1(const Mat &src, Mat &dst, Stream &stream) noexcept -> Status
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
    const auto srcData = static_cast<const Npp32f *>(src.Data());
    const auto dstData = static_cast<Npp32f *>(dst.Data());
    const auto srcStride = static_cast<int>(src.Stride());
    const auto dstStride = static_cast<int>(dst.Stride());
    const auto srcSize = NppiSize{static_cast<int>(src.Cols()), static_cast<int>(src.Rows())};
    const auto dstSize = NppiSize{static_cast<int>(dst.Cols()), static_cast<int>(dst.Rows())};
    const auto srcRoi = NppiRect{0, 0, srcSize.width, srcSize.height};
    const auto dstRoi = NppiRect{0, 0, dstSize.width, dstSize.height};

    NppStatus status = nppiResize_32f_C1R_Ctx(srcData, srcStride, //
                                              srcSize, srcRoi,    //
                                              dstData, dstStride, //
                                              dstSize, dstRoi,    //
                                              NPPI_INTER_NN, stream.GetNppStreamContext());

    if (status != NPP_SUCCESS)
    {
        LogError("nppiResize_32f_C1R failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    float value_scale = static_cast<float>(dst.Cols()) / static_cast<float>(src.Cols());
    status = nppiMulC_32f_C1IR_Ctx(static_cast<Npp32f>(value_scale),                                  //
                                   static_cast<Npp32f *>(dst.Data()), static_cast<int>(dst.Stride()), //
                                   dstSize,                                                           //
                                   stream.GetNppStreamContext());

    if (status != NPP_SUCCESS)
    {
        LogError("nppiMulC_32f_C1IR failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    return Status{};
#else
    (void)src;
    (void)dst;
    (void)stream;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto ConvertImage8UToC1(const Mat &src, Mat &dst, Stream &stream) noexcept -> Status
{
    if (src.Empty() || dst.Empty())
    {
        LogError("Source or destination is empty.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (dst.Channels() != 1)
    {
        LogError("Destination must have 1 channel.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (src.Channels() != 1 && src.Channels() != 3)
    {
        LogError("Source must have 1 or 3 channels.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if ((src.Cols() != dst.Cols()) || (src.Rows() != dst.Rows()))
    {
        LogError("Source and destination must have the same size.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    const auto srcData = static_cast<const Npp8u *>(src.Data());
    const auto dstData = static_cast<Npp8u *>(dst.Data());
    const auto srcStride = static_cast<int>(src.Stride());
    const auto dstStride = static_cast<int>(dst.Stride());
    const auto srcSize = NppiSize{static_cast<int>(src.Cols()), static_cast<int>(src.Rows())};

    if (src.Channels() == 1)
    {
        NppStatus status = nppiCopy_8u_C1R_Ctx(srcData, srcStride, //
                                               dstData, dstStride, //
                                               srcSize,            //
                                               stream.GetNppStreamContext());

        if (status != NPP_SUCCESS)
        {
            LogError("nppiCopy_8u_C1R failed");
            return Status{StatusCategory::CUDA, StatusCode::FAIL};
        }

        return Status{};
    }

    NppStatus status = nppiRGBToGray_8u_C3C1R_Ctx(srcData, srcStride, //
                                                  dstData, dstStride, //
                                                  srcSize,            //
                                                  stream.GetNppStreamContext());

    if (status != NPP_SUCCESS)
    {
        LogError("nppiRGBToGray_8u_C3C1R failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    };

    return Status{};
#else
    (void)src;
    (void)dst;
    (void)stream;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto Convert8UC1To32FC1(const Mat &src, Mat &dst, Stream &stream) noexcept -> Status
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

    if ((src.Cols() != dst.Cols()) || (src.Rows() != dst.Rows()))
    {
        LogError("Source and destination must have the same size.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    const auto srcData = static_cast<const Npp8u *>(src.Data());
    const auto dstData = static_cast<Npp32f *>(dst.Data());
    const auto srcStride = static_cast<int>(src.Stride());
    const auto dstStride = static_cast<int>(dst.Stride());
    const auto srcSize = NppiSize{static_cast<int>(src.Cols()), static_cast<int>(src.Rows())};

    NppStatus status = nppiConvert_8u32f_C1R_Ctx(srcData, srcStride, //
                                                 dstData, dstStride, //
                                                 srcSize,            //
                                                 stream.GetNppStreamContext());

    if (status != NPP_SUCCESS)
    {
        LogError("nppiConvert_8u32f_C1R failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    };

    return Status{};
#else
    (void)src;
    (void)dst;
    (void)stream;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto Convert8UC3To32FC3(const Mat &src, Mat &dst, Stream &stream) noexcept -> Status
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

    if ((src.Cols() != dst.Cols()) || (src.Rows() != dst.Rows()))
    {
        LogError("Source and destination must have the same size.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    const auto srcData = static_cast<const Npp8u *>(src.Data());
    const auto dstData = static_cast<Npp32f *>(dst.Data());
    const auto srcStride = static_cast<int>(src.Stride());
    const auto dstStride = static_cast<int>(dst.Stride());
    const auto srcSize = NppiSize{static_cast<int>(src.Cols()), static_cast<int>(src.Rows())};

    NppStatus status = nppiConvert_8u32f_C3R_Ctx(srcData, srcStride, //
                                                 dstData, dstStride, //
                                                 srcSize,            //
                                                 stream.GetNppStreamContext());

    if (status != NPP_SUCCESS)
    {
        LogError("nppiConvert_8u32f_C3R failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    };

    return Status{};
#else
    (void)src;
    (void)dst;
    (void)stream;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto DisparityOcclusionFilter32FC1(const Mat &src, Mat &dst, Stream &stream) noexcept -> Status
{
    if (src.Empty() || dst.Empty())
    {
        LogError("One of the input or output is empty.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (src.Channels() != 1 || dst.Channels() != 1)
    {
        LogError("All of the input and output must have 1 channel.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if ((src.Cols() != dst.Cols()) || (src.Rows() != dst.Rows()))
    {
        LogError("Input and output must have the same size.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    cudaError_t error = cudaDisparityOcclusionFilter(static_cast<const float *>(src.Data()), src.Stride(), //
                                                     static_cast<float *>(dst.Data()), dst.Stride(),       //
                                                     static_cast<std::uint32_t>(src.Cols()),               //
                                                     static_cast<std::uint32_t>(src.Rows()),               //
                                                     stream.GetCudaStream());

    if (error != cudaSuccess)
    {
        LogError("cudaDisparityOcclusionFilter failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    return Status{};
#else
    (void)src;
    (void)dst;
    (void)stream;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto HorizontalFlip8UC3(const Mat &src, Mat &dst, Stream &stream) noexcept -> Status
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
    NppStatus status = nppiMirror_8u_C3R_Ctx(static_cast<const Npp8u *>(src.Data()), static_cast<int>(src.Stride()), //
                                             static_cast<Npp8u *>(dst.Data()), static_cast<int>(dst.Stride()),       //
                                             {static_cast<int>(src.Cols()), static_cast<int>(src.Rows())},           //
                                             NPP_VERTICAL_AXIS, stream.GetNppStreamContext());

    if (status != NPP_SUCCESS)
    {
        LogError("nppiMirror_8u_C3R failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    };

    return Status{};
#else
    (void)src;
    (void)dst;
    (void)stream;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto HorizontalFlip32FC1(const Mat &src, Mat &dst, Stream &stream) noexcept -> Status
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
    NppStatus status = nppiMirror_32f_C1R_Ctx(static_cast<const Npp32f *>(src.Data()), static_cast<int>(src.Stride()), //
                                              static_cast<Npp32f *>(dst.Data()), static_cast<int>(dst.Stride()),       //
                                              {static_cast<int>(src.Cols()), static_cast<int>(src.Rows())},            //
                                              NPP_VERTICAL_AXIS, stream.GetNppStreamContext());

    if (status != NPP_SUCCESS)
    {
        LogError("nppiMirror_32f_C1R failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    };

    return Status{};
#else
    (void)src;
    (void)dst;
    (void)stream;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto LRConsistencyCheck32FC1(const Mat &left, const Mat &right, Mat &output, float relativeError, Stream &stream) noexcept -> Status
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
                                               relativeError, stream.GetCudaStream());

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
    (void)stream;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}
} // namespace retinify
