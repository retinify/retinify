// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/attributes.hpp"
#include "retinify/status.hpp"

#ifdef BUILD_WITH_TENSORRT
#include <cuda_runtime.h>
#include <npp.h>
#include <nppcore.h>
#else
#endif

namespace retinify
{
class RETINIFY_API Stream
{
  public:
    Stream() noexcept = default;
    ~Stream() noexcept;
    Stream(const Stream &) = delete;
    auto operator=(const Stream &) noexcept -> Stream & = delete;
    Stream(Stream &&) noexcept = delete;
    auto operator=(Stream &&) noexcept -> Stream & = delete;
    [[nodiscard]] auto Create() noexcept -> Status;
    [[nodiscard]] auto Destroy() noexcept -> Status;
    [[nodiscard]] auto Synchronize() const noexcept -> Status;

#ifdef BUILD_WITH_TENSORRT
    [[nodiscard]] auto GetCudaStream() const noexcept -> cudaStream_t;
    [[nodiscard]] auto GetNppStreamContext() const noexcept -> NppStreamContext;
#endif

  private:
#ifdef BUILD_WITH_TENSORRT
    cudaStream_t stream_{nullptr};
    NppStreamContext ctx_{nullptr};
#endif
};
} // namespace retinify
