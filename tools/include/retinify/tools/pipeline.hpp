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
enum class Resolution : std::uint8_t
{
    SMALL,  // height=320, width=640
    MEDIUM, // height=480, width=640
    LARGE,  // height=720, width=1280
};

/// @brief A class that wraps retinify::Pipeline to perform left-right consistency checks, providing an OpenCV-compatible interface for stereo matching.
class RETINIFY_API LRConsistencyPipeline
{
  public:
    LRConsistencyPipeline() noexcept = default;
    ~LRConsistencyPipeline() noexcept = default;
    LRConsistencyPipeline(const LRConsistencyPipeline &) = delete;
    auto operator=(const LRConsistencyPipeline &) noexcept -> LRConsistencyPipeline & = delete;
    LRConsistencyPipeline(LRConsistencyPipeline &&) = delete;
    auto operator=(LRConsistencyPipeline &&other) noexcept -> LRConsistencyPipeline & = delete;

    /// @brief Initializes the stereo matching pipeline with the processing resolution.
    /// @param resolution
    /// The processing resolution for the stereo matching pipeline.
    /// @return
    /// A Status object indicating the success or failure of the initialization.
    [[nodiscard]] auto Initialize(Resolution resolution = Resolution::LARGE) noexcept -> Status;

    /// @brief Runs the stereo matching pipeline with left-right consistency check using the provided left and right image data.
    /// @param leftImage
    /// The left image data as a `cv::Mat`.
    /// @param rightImage
    /// The right image data as a `cv::Mat`.
    /// @param disparity
    /// The output disparity data as a `cv::Mat`.
    /// @param maxDisparityDifference
    /// The maximum allowed difference in disparity values for the left-right consistency check.
    /// @return
    /// A Status object indicating the success or failure of the operation.
    [[nodiscard]] auto Run(const cv::Mat &leftImage, const cv::Mat &rightImage, cv::Mat &disparity, //
                           float maxDisparityDifference = 1.0F) const noexcept -> Status;

  private:
    size_t imageHeight_{0};
    size_t imageWidth_{0};
    Pipeline pipeline_;
};
} // namespace retinify::tools