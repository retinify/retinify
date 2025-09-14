// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mat.hpp"

#include "retinify/logging.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>

namespace retinify
{
Mat::~Mat() noexcept
{
    (void)this->Free();
}

auto Mat::Allocate(std::size_t rows, std::size_t cols, std::size_t channels, std::size_t bytesPerElement) noexcept -> Status
{
    Status status = this->Free();
    if (!status.IsOK())
    {
        return status;
    }

    if (rows == 0 || cols == 0 || channels == 0 || bytesPerElement == 0)
    {
        LogError("Not allowed zero dimensions.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    if (channels > std::numeric_limits<std::size_t>::max() / cols)
    {
        LogError("Overflow in channels * cols.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    std::size_t elementsPerRow = cols * channels;
    if (elementsPerRow > std::numeric_limits<std::size_t>::max() / bytesPerElement)
    {
        LogError("Overflow in elementsPerRow * bytesPerElement.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    std::size_t columnsInBytes = elementsPerRow * bytesPerElement;
    if (rows > std::numeric_limits<std::size_t>::max() / columnsInBytes)
    {
        LogError("Overflow in rows * columnsInBytes.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    void *newDeviceData = nullptr;
    std::size_t newStride = 0;

#ifdef BUILD_WITH_TENSORRT
    cudaError_t streamError = cudaStreamCreate(&stream_);
    if (streamError != cudaSuccess)
    {
        LogError(cudaGetErrorString(streamError));
        return Status(StatusCategory::CUDA, StatusCode::FAIL);
    }

    cudaError_t mallocError = cudaMallocPitch(&newDeviceData, &newStride, columnsInBytes, rows);
    if (mallocError != cudaSuccess)
    {
        LogError(cudaGetErrorString(mallocError));
        return Status(StatusCategory::CUDA, StatusCode::FAIL);
    }

    if (rows > std::numeric_limits<std::size_t>::max() / newStride)
    {
        if (newDeviceData != nullptr)
        {
            cudaError_t freeError = cudaFree(newDeviceData);
            if (freeError != cudaSuccess)
            {
                LogError(cudaGetErrorString(freeError));
            }
            newDeviceData = nullptr;
        }
        LogError("Overflow in rows * newStride.");
        return Status(StatusCategory::CUDA, StatusCode::FAIL);
    }
#else
    constexpr std::size_t alignment = 64;
    if (columnsInBytes > std::numeric_limits<std::size_t>::max() - (alignment - 1))
    {
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    newStride = ((columnsInBytes + alignment - 1) / alignment) * alignment;
    if (rows > std::numeric_limits<std::size_t>::max() / newStride)
    {
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    std::size_t allocSize = newStride * rows;
    if (allocSize > std::numeric_limits<std::size_t>::max() - (alignment - 1))
    {
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    std::size_t alignedSize = ((allocSize + alignment - 1) / alignment) * alignment;
    newDeviceData = std::aligned_alloc(alignment, alignedSize);
    if (newDeviceData == nullptr)
    {
        return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
    }
#endif

    this->deviceData_ = newDeviceData;
    this->deviceStride_ = newStride;
    this->rows_ = rows;
    this->cols_ = cols;
    this->channels_ = channels;
    this->bytesPerElement_ = bytesPerElement;
    this->deviceRows_ = rows;
    this->deviceColumnsInBytes_ = columnsInBytes;

    return Status{};
}

auto Mat::Free() noexcept -> Status
{
    Status status;

    if (deviceData_ != nullptr)
    {
#ifdef BUILD_WITH_TENSORRT
        cudaError_t freeError = cudaFree(deviceData_);
        if (freeError != cudaSuccess)
        {
            LogError(cudaGetErrorString(freeError));
            status = Status(StatusCategory::CUDA, StatusCode::FAIL);
        }
#else
        std::free(deviceData_);
#endif
        this->deviceData_ = nullptr;
    }

#ifdef BUILD_WITH_TENSORRT
    if (stream_ != nullptr)
    {
        cudaError_t streamError = cudaStreamDestroy(stream_);
        if (streamError != cudaSuccess)
        {
            LogError(cudaGetErrorString(streamError));
            status = Status(StatusCategory::CUDA, StatusCode::FAIL);
        }
        stream_ = nullptr;
    }

#endif

    this->deviceStride_ = 0;
    this->rows_ = 0;
    this->cols_ = 0;
    this->channels_ = 0;
    this->bytesPerElement_ = 0;
    this->deviceRows_ = 0;
    this->deviceColumnsInBytes_ = 0;

    return status;
}

auto Mat::Upload(const void *hostData, std::size_t hostStride) const noexcept -> Status
{
    if (deviceData_ == nullptr)
    {
        LogError("Device data is not allocated.");
        return Status(StatusCategory::RETINIFY, StatusCode::FAIL);
    }

    if (hostData == nullptr)
    {
        LogError("Host data pointer is null.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    if (hostStride < deviceColumnsInBytes_)
    {
        LogError("Host stride is less than device columns in bytes.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

#ifdef BUILD_WITH_TENSORRT
    cudaError_t copyError = cudaMemcpy2DAsync(deviceData_, deviceStride_, hostData, hostStride, deviceColumnsInBytes_, deviceRows_, cudaMemcpyHostToDevice, stream_);
    if (copyError != cudaSuccess)
    {
        LogError(cudaGetErrorString(copyError));
        return Status(StatusCategory::CUDA, StatusCode::FAIL);
    }

#else
    const unsigned char *src = static_cast<const unsigned char *>(hostData);
    unsigned char *dst = static_cast<unsigned char *>(deviceData_);
    for (std::size_t r = 0; r < deviceRows_; ++r)
    {
        std::memcpy(dst + r * deviceStride_, src + r * hostStride, deviceColumnsInBytes_);
    }
#endif

    return Status{};
}

auto Mat::Download(void *hostData, std::size_t hostStride) const noexcept -> Status
{
    if (deviceData_ == nullptr)
    {
        LogError("Device data is not allocated.");
        return Status(StatusCategory::RETINIFY, StatusCode::FAIL);
    }

    if (hostData == nullptr)
    {
        LogError("Host data pointer is null.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    if (hostStride < deviceColumnsInBytes_)
    {
        LogError("Host stride is less than device columns in bytes.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

#ifdef BUILD_WITH_TENSORRT
    cudaError_t copyError = cudaMemcpy2DAsync(hostData, hostStride, deviceData_, deviceStride_, deviceColumnsInBytes_, deviceRows_, cudaMemcpyDeviceToHost, stream_);
    if (copyError != cudaSuccess)
    {
        LogError(cudaGetErrorString(copyError));
        return Status(StatusCategory::CUDA, StatusCode::FAIL);
    }

#else
    const unsigned char *src = static_cast<const unsigned char *>(deviceData_);
    unsigned char *dst = static_cast<unsigned char *>(hostData);
    for (std::size_t r = 0; r < deviceRows_; ++r)
    {
        std::memcpy(dst + r * hostStride, src + r * deviceStride_, deviceColumnsInBytes_);
    }
#endif

    return Status{};
}

auto Mat::Wait() const noexcept -> Status
{
#ifdef BUILD_WITH_TENSORRT
    if (stream_ != nullptr)
    {
        cudaError_t waitError = cudaStreamSynchronize(stream_);
        if (waitError != cudaSuccess)
        {
            LogError(cudaGetErrorString(waitError));
            return Status(StatusCategory::CUDA, StatusCode::FAIL);
        }
    }
#endif

    return Status{};
}

auto Mat::Data() const noexcept -> void *
{
    return deviceData_;
}

auto Mat::Empty() const noexcept -> bool
{
    return deviceData_ == nullptr;
}

auto Mat::Rows() const noexcept -> std::size_t
{
    return rows_;
}

auto Mat::Cols() const noexcept -> std::size_t
{
    return cols_;
}

auto Mat::Channels() const noexcept -> std::size_t
{
    return channels_;
}

auto Mat::BytesPerElement() const noexcept -> std::size_t
{
    return bytesPerElement_;
}

auto Mat::ElementCount() const noexcept -> std::size_t
{
    return rows_ * cols_ * channels_;
}

auto Mat::Stride() const noexcept -> std::size_t
{
    return deviceStride_;
}

auto Mat::Shape() const noexcept -> std::array<int64_t, 4>
{
    return {1, static_cast<int64_t>(rows_), static_cast<int64_t>(cols_), static_cast<int64_t>(channels_)};
}
} // namespace retinify