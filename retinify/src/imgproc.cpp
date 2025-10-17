// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-eula

#include "imgproc.hpp"

#include "retinify/logging.hpp"

#ifdef BUILD_WITH_TENSORRT
#include "cuda/depth.h"
#include "cuda/lrcheck.h"
#include "cuda/occlusion.h"
#include "cuda/reproject.h"
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

auto RemapImage8U(const Mat &src, const Mat &mapX, const Mat &mapY, Mat &dst, Stream &stream) noexcept -> Status
{
    if (src.Empty() || mapX.Empty() || mapY.Empty() || dst.Empty())
    {
        LogError("Source, maps, or destination is empty.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (src.BytesPerElement() != sizeof(std::uint8_t) || dst.BytesPerElement() != sizeof(std::uint8_t))
    {
        LogError("Source and destination must be 8-bit unsigned.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (mapX.BytesPerElement() != sizeof(float) || mapY.BytesPerElement() != sizeof(float))
    {
        LogError("Maps must be 32-bit floating-point.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (src.Channels() != dst.Channels())
    {
        LogError("Source and destination must have the same number of channels.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (mapX.Channels() != 1 || mapY.Channels() != 1)
    {
        LogError("Maps must have 1 channel.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if ((dst.Cols() != mapX.Cols()) || (dst.Rows() != mapX.Rows()) || (dst.Cols() != mapY.Cols()) || (dst.Rows() != mapY.Rows()))
    {
        LogError("Destination and maps must have the same size.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if ((src.Cols() == 0) || (src.Rows() == 0))
    {
        LogError("Source size must be non-zero.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (src.Channels() != 1 && src.Channels() != 3)
    {
        LogError("Source and destination must have 1 or 3 channels.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    const auto *srcData = static_cast<const Npp8u *>(src.Data());
    auto *dstData = static_cast<Npp8u *>(dst.Data());
    const auto *mapXData = static_cast<const Npp32f *>(mapX.Data());
    const auto *mapYData = static_cast<const Npp32f *>(mapY.Data());
    const auto srcSize = NppiSize{static_cast<int>(src.Cols()), static_cast<int>(src.Rows())};
    const auto dstSize = NppiSize{static_cast<int>(dst.Cols()), static_cast<int>(dst.Rows())};
    const auto srcRoi = NppiRect{0, 0, srcSize.width, srcSize.height};
    const auto srcStride = static_cast<int>(src.Stride());
    const auto dstStride = static_cast<int>(dst.Stride());
    const auto mapXStep = static_cast<int>(mapX.Stride());
    const auto mapYStep = static_cast<int>(mapY.Stride());

    NppStatus status{};

    if (src.Channels() == 1)
    {
        status = nppiRemap_8u_C1R_Ctx(srcData, srcSize, srcStride, srcRoi,    //
                                      mapXData, mapXStep, mapYData, mapYStep, //
                                      dstData, dstStride, dstSize,            //
                                      NPPI_INTER_LINEAR, stream.GetNppStreamContext());

        if (status != NPP_SUCCESS)
        {
            LogError("nppiRemap_8u_C1R failed");
            return Status{StatusCategory::CUDA, StatusCode::FAIL};
        }

        return Status{};
    }

    status = nppiRemap_8u_C3R_Ctx(srcData, srcSize, srcStride, srcRoi,                                 //
                                  mapXData, mapXStep, mapYData, mapYStep, dstData, dstStride, dstSize, //
                                  NPPI_INTER_LINEAR, stream.GetNppStreamContext());

    if (status != NPP_SUCCESS)
    {
        LogError("nppiRemap_8u_C3R failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    return Status{};
#else
    (void)src;
    (void)mapX;
    (void)mapY;
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

auto ReprojectDisparityTo3D(const Mat &disparity, Mat &points3d, const Mat4x4d &Q, Stream &stream) noexcept -> Status
{
    if (disparity.Empty() || points3d.Empty())
    {
        LogError("Disparity or output points buffer is empty.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (disparity.Channels() != 1 || points3d.Channels() != 3)
    {
        LogError("Disparity must have 1 channel and points must have 3 channels.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (disparity.BytesPerElement() != sizeof(float) || points3d.BytesPerElement() != sizeof(float))
    {
        LogError("Disparity and points must have 32-bit floating-point elements.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (disparity.Rows() != points3d.Rows() || disparity.Cols() != points3d.Cols())
    {
        LogError("Disparity and points must have the same spatial dimensions.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    float q[16];
    for (std::size_t row = 0; row < 4; ++row)
    {
        for (std::size_t col = 0; col < 4; ++col)
        {
            q[row * 4 + col] = static_cast<float>(Q[row][col]);
        }
    }

    cudaError_t error = cudaReprojectTo3d(static_cast<const float *>(disparity.Data()), disparity.Stride(), //
                                          static_cast<float *>(points3d.Data()), points3d.Stride(),         //
                                          static_cast<std::uint32_t>(disparity.Cols()),                     //
                                          static_cast<std::uint32_t>(disparity.Rows()),                     //
                                          q,                                                                //
                                          stream.GetCudaStream());

    if (error != cudaSuccess)
    {
        LogError("cudaReprojectTo3d failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    return Status{};
#else
    (void)disparity;
    (void)points3d;
    (void)Q;
    (void)stream;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto DisparityToDepth32FC1(const Mat &disparity, Mat &depth, const Mat4x4d &Q, Stream &stream) noexcept -> Status
{
    if (disparity.Empty() || depth.Empty())
    {
        LogError("Disparity or depth buffer is empty.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (disparity.Channels() != 1 || depth.Channels() != 1)
    {
        LogError("Disparity and depth must have exactly 1 channel.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (disparity.BytesPerElement() != sizeof(float) || depth.BytesPerElement() != sizeof(float))
    {
        LogError("Disparity and depth must have 32-bit floating-point elements.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    if (disparity.Rows() != depth.Rows() || disparity.Cols() != depth.Cols())
    {
        LogError("Disparity and depth must have the same spatial dimensions.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

#ifdef BUILD_WITH_TENSORRT
    float q[16];
    for (std::size_t row = 0; row < 4; ++row)
    {
        for (std::size_t col = 0; col < 4; ++col)
        {
            q[row * 4 + col] = static_cast<float>(Q[row][col]);
        }
    }

    cudaError_t error = cudaDisparityToDepth(static_cast<const float *>(disparity.Data()), disparity.Stride(), //
                                             static_cast<float *>(depth.Data()), depth.Stride(),               //
                                             static_cast<std::uint32_t>(disparity.Cols()),                     //
                                             static_cast<std::uint32_t>(disparity.Rows()),                     //
                                             q,                                                                //
                                             stream.GetCudaStream());

    if (error != cudaSuccess)
    {
        LogError("cudaDisparityToDepth failed");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    return Status{};
#else
    (void)disparity;
    (void)depth;
    (void)Q;
    (void)stream;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}
} // namespace retinify
