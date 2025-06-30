// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mat.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace retinify
{
Mat::Mat() noexcept
{
#ifdef USE_NVIDIA_GPU
    cudaError_t streamErr = cudaStreamCreate(&stream_);
    if (streamErr != cudaSuccess)
    {
        stream_ = nullptr;
    }

    cudaError_t eventErr = cudaEventCreate(&event_);
    if (eventErr != cudaSuccess)
    {
        event_ = nullptr;
    }
#endif
}

Mat::~Mat() noexcept
{
    (void)this->Free();

#ifdef USE_NVIDIA_GPU
    if (stream_ != nullptr)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    if (event_ != nullptr)
    {
        cudaEventDestroy(event_);
        event_ = nullptr;
    }
#endif
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
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    if (channels > std::numeric_limits<std::size_t>::max() / cols)
    {
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    std::size_t elementsPerRow = cols * channels;
    if (elementsPerRow > std::numeric_limits<std::size_t>::max() / bytesPerElement)
    {
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    std::size_t columnsInBytes = elementsPerRow * bytesPerElement;
    if (rows > std::numeric_limits<std::size_t>::max() / columnsInBytes)
    {
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    void *newDeviceData = nullptr;
    std::size_t newPitch = 0;

#ifdef USE_NVIDIA_GPU
    cudaError_t err = cudaMallocPitch(&newDeviceData, &newPitch, columnsInBytes, rows);
    if (err != cudaSuccess)
    {
        return Status(StatusCategory::CUDA, StatusCode::FAIL);
    }

    if (rows > std::numeric_limits<std::size_t>::max() / newPitch)
    {
        cudaFree(newDeviceData);
        return Status(StatusCategory::CUDA, StatusCode::FAIL);
    }
#else
    constexpr std::size_t alignment = 64;
    if (columnsInBytes > std::numeric_limits<std::size_t>::max() - (alignment - 1))
    {
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }
    newPitch = ((columnsInBytes + alignment - 1) / alignment) * alignment;
    if (rows > std::numeric_limits<std::size_t>::max() / newPitch)
    {
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }
    std::size_t allocSize = newPitch * rows;
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
    this->devicePitch_ = newPitch;
    this->rows_ = rows;
    this->cols_ = cols;
    this->channels_ = channels;
    this->bytesPerElement_ = bytesPerElement;
    this->deviceRows_ = rows;
    this->deviceColumnsInBytes_ = columnsInBytes;

    return status;
}

auto Mat::Free() noexcept -> Status
{
    Status status;

    if (deviceData_ != nullptr)
    {
#ifdef USE_NVIDIA_GPU
        cudaError_t err = cudaFree(deviceData_);
        if (err != cudaSuccess)
        {
            return Status(StatusCategory::CUDA, StatusCode::FAIL);
        }
#else
        std::free(deviceData_);
#endif
    }

    this->deviceData_ = nullptr;
    this->devicePitch_ = 0;
    this->rows_ = 0;
    this->cols_ = 0;
    this->channels_ = 0;
    this->bytesPerElement_ = 0;
    this->deviceRows_ = 0;
    this->deviceColumnsInBytes_ = 0;

    return status;
}

auto Mat::Upload(const void *hostData, std::size_t hostPitch) const noexcept -> Status
{
    if (deviceData_ == nullptr)
    {
        return Status(StatusCategory::USER, StatusCode::NOT_ALLOCATED);
    }

    if (hostData == nullptr)
    {
        return Status(StatusCategory::USER, StatusCode::NULL_POINTER);
    }

    if (hostPitch < deviceColumnsInBytes_)
    {
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

#ifdef USE_NVIDIA_GPU
    cudaError_t copyErr = cudaMemcpy2DAsync(deviceData_, devicePitch_, hostData, hostPitch, deviceColumnsInBytes_, deviceRows_, cudaMemcpyHostToDevice, stream_);
    if (copyErr != cudaSuccess)
    {
        return Status(StatusCategory::CUDA, StatusCode::FAIL);
    }

    cudaError_t eventErr = cudaEventRecord(event_, stream_);
    if (eventErr != cudaSuccess)
    {
        return Status(StatusCategory::CUDA, StatusCode::FAIL);
    }
#else
    const unsigned char *src = static_cast<const unsigned char *>(hostData);
    unsigned char *dst = static_cast<unsigned char *>(deviceData_);
    for (std::size_t r = 0; r < deviceRows_; ++r)
    {
        std::memcpy(dst + r * devicePitch_, src + r * hostPitch, deviceColumnsInBytes_);
    }
#endif

    return Status{};
}

auto Mat::Download(void *hostData, std::size_t hostPitch) const noexcept -> Status
{
    if (deviceData_ == nullptr)
    {
        return Status(StatusCategory::USER, StatusCode::NOT_ALLOCATED);
    }

    if (hostData == nullptr)
    {
        return Status(StatusCategory::USER, StatusCode::NULL_POINTER);
    }

    if (hostPitch < deviceColumnsInBytes_)
    {
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

#ifdef USE_NVIDIA_GPU
    cudaError_t copyErr = cudaMemcpy2DAsync(hostData, hostPitch, deviceData_, devicePitch_, deviceColumnsInBytes_, deviceRows_, cudaMemcpyDeviceToHost, stream_);
    if (copyErr != cudaSuccess)
    {
        return Status(StatusCategory::CUDA, StatusCode::FAIL);
    }

    cudaError_t eventErr = cudaEventRecord(event_, stream_);
    if (eventErr != cudaSuccess)
    {
        return Status(StatusCategory::CUDA, StatusCode::FAIL);
    }
#else
    const unsigned char *src = static_cast<const unsigned char *>(deviceData_);
    unsigned char *dst = static_cast<unsigned char *>(hostData);
    for (std::size_t r = 0; r < deviceRows_; ++r)
    {
        std::memcpy(dst + r * hostPitch, src + r * devicePitch_, deviceColumnsInBytes_);
    }
#endif

    return Status{};
}

auto Mat::Wait() const noexcept -> Status
{
#ifdef USE_NVIDIA_GPU
    if (stream_ != nullptr)
    {
        cudaError_t err = cudaStreamWaitEvent(stream_, event_, 0);
        if (err != cudaSuccess)
        {
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

auto Mat::Shape() const noexcept -> std::array<int64_t, 4>
{
    return {1, static_cast<int64_t>(rows_), static_cast<int64_t>(cols_), static_cast<int64_t>(channels_)};
}
} // namespace retinify