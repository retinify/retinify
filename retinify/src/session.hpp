// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mat.hpp"
#include "stream.hpp"

#include "retinify/status.hpp"

#include <array>
#include <memory>

#ifdef BUILD_WITH_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <iostream>
#else
#endif

namespace retinify
{
class RETINIFY_API Session
{
  public:
    Session() noexcept = default;
    ~Session() noexcept;
    Session(const Session &) = delete;
    auto operator=(const Session &) noexcept -> Session & = delete;
    Session(Session &&other) noexcept = delete;
    auto operator=(Session &&other) noexcept -> Session & = delete;
    [[nodiscard]] auto Initialize(const char *model_path) noexcept -> Status;
    [[nodiscard]] auto BindInput(const char *name, const Mat &mat) const noexcept -> Status;
    [[nodiscard]] auto BindOutput(const char *name, const Mat &mat) const noexcept -> Status;
    [[nodiscard]] auto Run(Stream &stream) const noexcept -> Status;

  private:
#ifdef BUILD_WITH_TENSORRT
    nvinfer1::IRuntime *runtime_{nullptr};
    nvinfer1::ICudaEngine *engine_{nullptr};
    nvinfer1::IExecutionContext *context_{nullptr};
#else
#endif
};
} // namespace retinify