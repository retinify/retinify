// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/define.hpp"
#include "retinify/pipeline.hpp"
#include "retinify/status.hpp"

#include <opencv2/core.hpp>

namespace retinify
{
namespace tools
{
class RETINIFY_API LRConsistencyPipeline
{
  public:
    LRConsistencyPipeline() = default;
    ~LRConsistencyPipeline() = default;
    [[nodiscard]] Status Initialize(std::size_t imageHeight, std::size_t imageWidth) noexcept;
    [[nodiscard]] Status Run(const cv::Mat &leftImage, const cv::Mat &rightImage, cv::Mat &disparity) const noexcept;

  private:
    Pipeline pipeline_;
};
} // namespace tools
} // namespace retinify