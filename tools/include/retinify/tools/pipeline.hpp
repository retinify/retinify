// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/define.hpp"
#include "retinify/pipeline.hpp"
#include "retinify/status.hpp"

#include <opencv2/core.hpp>

namespace retinify::tools
{
/// @brief The resolution options for stereo matching pipelines.
enum class PipelineResolution : std::uint8_t
{
    LOW,    // height=320, width=640
    MEDIUM, // height=480, width=640
    HIGH,   // height=720, width=1280
};

class RETINIFY_API LRConsistencyPipeline
{
  public:
    LRConsistencyPipeline() = default;
    ~LRConsistencyPipeline() = default;
    LRConsistencyPipeline(const LRConsistencyPipeline &) = delete;
    auto operator=(const LRConsistencyPipeline &) noexcept -> LRConsistencyPipeline & = delete;
    LRConsistencyPipeline(LRConsistencyPipeline &&) = delete;
    auto operator=(LRConsistencyPipeline &&other) noexcept -> LRConsistencyPipeline & = delete;
    [[nodiscard]] auto Initialize(PipelineResolution resolution = PipelineResolution::HIGH) noexcept -> Status;
    [[nodiscard]] auto Run(const cv::Mat &leftImage, const cv::Mat &rightImage, cv::Mat &disparity, //
                           float maxDisparityDifference = 1.0F) const noexcept -> Status;

  private:
    cv::Size imageSize_;
    Pipeline pipeline_;
};
} // namespace retinify::tools