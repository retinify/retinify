// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "stream.hpp"

#include "retinify/logging.hpp"

namespace retinify
{
Stream::~Stream() noexcept
{
    (void)Destroy();
}

auto Stream::Create() noexcept -> Status
{
    // if already created, destroy first
    auto statusDestroy = Destroy();
    if (!statusDestroy.IsOK())
    {
        return statusDestroy;
    }

#ifdef BUILD_WITH_TENSORRT
    cudaError_t cudaStreamStatus = cudaStreamCreate(&stream_);
    if (cudaStreamStatus != cudaSuccess)
    {
        LogError(cudaGetErrorString(cudaStreamStatus));
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    ctx_.hStream = stream_;

    LogDebug("Created cudaStream_t.");
    return Status{};
#else
#endif
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
}

auto Stream::Destroy() noexcept -> Status
{
#ifdef BUILD_WITH_TENSORRT
    if (stream_ != nullptr)
    {
        cudaError_t streamError = cudaStreamDestroy(stream_);
        if (streamError != cudaSuccess)
        {
            LogError(cudaGetErrorString(streamError));
            return Status{StatusCategory::CUDA, StatusCode::FAIL};
        }
        stream_ = nullptr;

        LogDebug("Destroyed cudaStream_t.");
    }
    return Status{};
#else
#endif
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
}

auto Stream::Synchronize() const noexcept -> Status
{
#ifdef BUILD_WITH_TENSORRT
    if (stream_ != nullptr)
    {
        cudaError_t syncError = cudaStreamSynchronize(stream_);
        if (syncError != cudaSuccess)
        {
            LogError(cudaGetErrorString(syncError));
            return Status{StatusCategory::CUDA, StatusCode::FAIL};
        }
    }

    LogDebug("Synchronized cudaStream_t.");
    return Status{};
#else
#endif
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
}

#ifdef BUILD_WITH_TENSORRT
auto Stream::GetCudaStream() const noexcept -> cudaStream_t
{
    return stream_;
}

auto Stream::GetNppStreamContext() const noexcept -> NppStreamContext
{
    return ctx_;
}
#endif
} // namespace retinify
