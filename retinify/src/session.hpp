// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mat.hpp"
#include "stream.hpp"

#include "retinify/nocopymove.hpp"
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
namespace detail
{
constexpr const char kOnnxLeftInputName[] = "left";
constexpr const char kOnnxRightInputName[] = "right";
constexpr const char kOnnxDisparityOutputName[] = "disparity";
constexpr int kEngineMinHeight = 320;
constexpr int kEngineMinWidth = 640;
constexpr int kEngineOptHeight = 480;
constexpr int kEngineOptWidth = 640;
constexpr int kEngineMaxHeight = 720;
constexpr int kEngineMaxWidth = 1280;

class RETINIFY_API Session : public NoCopyMove
{
  public:
    Session() noexcept = default;
    ~Session() noexcept = default;
    [[nodiscard]] auto Initialize(const char *modelPath) noexcept -> Status;
    [[nodiscard]] auto BindInput(const char *name, const Mat &mat) const noexcept -> Status;
    [[nodiscard]] auto BindOutput(const char *name, const Mat &mat) const noexcept -> Status;
    [[nodiscard]] auto Execute(Stream &stream) const noexcept -> Status;

  private:
#ifdef BUILD_WITH_TENSORRT
    std::unique_ptr<nvinfer1::IRuntime> runtime_{};
    std::unique_ptr<nvinfer1::ICudaEngine> engine_{};
    std::unique_ptr<nvinfer1::IExecutionContext> context_{};
#else
#endif
};
} // namespace detail
} // namespace retinify
