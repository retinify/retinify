// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/attributes.hpp"
#include "retinify/nocopymove.hpp"
#include "retinify/status.hpp"

#ifdef BUILD_WITH_TENSORRT
#include <cuda_runtime.h>
#include <npp.h>
#include <nppcore.h>
#else
#endif

namespace retinify
{
namespace detail
{
class RETINIFY_API Stream : public NoCopyMove
{
  public:
    Stream() noexcept = default;
    ~Stream() noexcept;
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
    NppStreamContext ctx_{};
#endif
};
} // namespace detail
} // namespace retinify
