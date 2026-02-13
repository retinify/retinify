// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mat.hpp"

#include "retinify/logging.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>

namespace retinify
{
namespace detail
{
Mat::~Mat() noexcept
{
    (void)this->Free();
}

auto Mat::Allocate(std::size_t rows, std::size_t cols, std::size_t channels, std::size_t bytesPerElement, MatLocation location) noexcept -> Status
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
    auto allocateHostBuffer = [&]() -> Status {
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

        std::memset(newDeviceData, 0, alignedSize);
        return Status{};
    };

    switch (location)
    {
    case MatLocation::DEVICE: {
#ifdef BUILD_WITH_TENSORRT
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
        Status hostStatus = allocateHostBuffer();
        if (!hostStatus.IsOK())
        {
            return hostStatus;
        }
#endif
        break;
    }
    case MatLocation::HOST: {
        Status hostStatus = allocateHostBuffer();
        if (!hostStatus.IsOK())
        {
            return hostStatus;
        }
        break;
    }
    default: {
        LogError("Invalid MatLocation specified.");
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }
    }

    this->deviceData_ = newDeviceData;
    this->deviceStride_ = newStride;
    this->rows_ = rows;
    this->cols_ = cols;
    this->channels_ = channels;
    this->bytesPerElement_ = bytesPerElement;
    this->deviceRows_ = rows;
    this->deviceColumnsInBytes_ = columnsInBytes;
    this->location_ = location;

    return Status{};
}

auto Mat::Free() noexcept -> Status
{
    Status status;

    if (deviceData_ != nullptr)
    {
        switch (location_)
        {
        case MatLocation::DEVICE: {
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
            break;
        }
        case MatLocation::HOST: {
            std::free(deviceData_);
            break;
        }
        default: {
            LogError("Invalid MatLocation specified.");
            status = Status(StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT);
            break;
        }
        }

        this->deviceData_ = nullptr;
    }

    this->deviceStride_ = 0;
    this->rows_ = 0;
    this->cols_ = 0;
    this->channels_ = 0;
    this->bytesPerElement_ = 0;
    this->deviceRows_ = 0;
    this->deviceColumnsInBytes_ = 0;
    this->location_ = MatLocation::UNKNOWN;

    return status;
}

auto Mat::Upload(const void *hostData, std::size_t hostStride, Stream &stream) const noexcept -> Status
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
        LogStrideError(hostStride, deviceColumnsInBytes_);
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    auto copyFromHostBuffer = [&](const void *srcData, std::size_t srcStride) -> Status {
        const unsigned char *src = static_cast<const unsigned char *>(srcData);
        unsigned char *dst = static_cast<unsigned char *>(deviceData_);
        for (std::size_t r = 0; r < deviceRows_; ++r)
        {
            std::memcpy(dst + r * deviceStride_, src + r * srcStride, deviceColumnsInBytes_);
        }
        return Status{};
    };

    switch (location_)
    {
    case MatLocation::DEVICE: {
#ifdef BUILD_WITH_TENSORRT
        cudaError_t copyError = cudaMemcpy2DAsync(deviceData_, deviceStride_, hostData, hostStride, deviceColumnsInBytes_, deviceRows_, cudaMemcpyHostToDevice, stream.GetCudaStream());
        if (copyError != cudaSuccess)
        {
            LogError(cudaGetErrorString(copyError));
            return Status(StatusCategory::CUDA, StatusCode::FAIL);
        }
#else
        return copyFromHostBuffer(hostData, hostStride);
#endif
        break;
    }
    case MatLocation::HOST: {
        return copyFromHostBuffer(hostData, hostStride);
    }
    default: {
        LogError("Invalid MatLocation specified.");
        return Status(StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT);
    }
    }

    return Status{};
}

auto Mat::Download(void *hostData, std::size_t hostStride, Stream &stream) const noexcept -> Status
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
        LogStrideError(hostStride, deviceColumnsInBytes_);
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    auto copyToHostBuffer = [&](void *dstData, std::size_t dstStride) -> Status {
        const unsigned char *src = static_cast<const unsigned char *>(deviceData_);
        unsigned char *dst = static_cast<unsigned char *>(dstData);
        for (std::size_t r = 0; r < deviceRows_; ++r)
        {
            std::memcpy(dst + r * dstStride, src + r * deviceStride_, deviceColumnsInBytes_);
        }
        return Status{};
    };

    switch (location_)
    {
    case MatLocation::DEVICE: {
#ifdef BUILD_WITH_TENSORRT
        cudaError_t copyError = cudaMemcpy2DAsync(hostData, hostStride, deviceData_, deviceStride_, deviceColumnsInBytes_, deviceRows_, cudaMemcpyDeviceToHost, stream.GetCudaStream());
        if (copyError != cudaSuccess)
        {
            LogError(cudaGetErrorString(copyError));
            return Status(StatusCategory::CUDA, StatusCode::FAIL);
        }
#else
        return copyToHostBuffer(hostData, hostStride);
#endif
        break;
    }
    case MatLocation::HOST: {
        return copyToHostBuffer(hostData, hostStride);
    }
    default: {
        LogError("Invalid MatLocation specified.");
        return Status(StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT);
    }
    }

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

auto Mat::Location() const noexcept -> MatLocation
{
    return location_;
}
} // namespace detail
} // namespace retinify
