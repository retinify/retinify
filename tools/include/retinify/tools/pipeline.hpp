// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/define.hpp"
#include "retinify/pipeline.hpp"
#include "retinify/status.hpp"

#include <opencv2/core.hpp>

namespace retinify::tools
{
/// @brief
/// The resolution options for stereo matching pipelines.
enum class Resolution : std::uint8_t
{
    /// height=320, width=640
    SMALL,
    /// height=480, width=640
    MEDIUM,
    /// height=720, width=1280
    LARGE,
};

/// @brief
/// This class provides an interface for running a stereo matching pipeline using OpenCV
class RETINIFY_API StereoMatchingPipeline
{
  public:
    StereoMatchingPipeline() noexcept = default;
    ~StereoMatchingPipeline() noexcept = default;
    StereoMatchingPipeline(const StereoMatchingPipeline &) = delete;
    auto operator=(const StereoMatchingPipeline &) noexcept -> StereoMatchingPipeline & = delete;
    StereoMatchingPipeline(StereoMatchingPipeline &&) = delete;
    auto operator=(StereoMatchingPipeline &&other) noexcept -> StereoMatchingPipeline & = delete;

    /// @brief
    /// Initializes the stereo matching pipeline with the specified processing resolution.
    /// @param resolution
    /// The processing resolution to use for the stereo matching pipeline.
    /// @return
    /// A Status object indicating whether initialization succeeded.
    [[nodiscard]] auto Initialize(Resolution resolution = Resolution::LARGE) noexcept -> Status;

    /// @brief
    /// Runs the stereo matching pipeline.
    /// @param leftImage
    /// The left input image as a `cv::Mat`.
    /// @param rightImage
    /// The right input image as a `cv::Mat`.
    /// @param disparity
    /// The output disparity map as a `cv::Mat`.
    /// @return
    /// A Status object indicating whether the operation succeeded.
    [[nodiscard]] auto Run(const cv::Mat &leftImage, const cv::Mat &rightImage, cv::Mat &disparity) const noexcept -> Status;

    /// @brief
    /// Runs the stereo matching pipeline with left-right consistency check on the provided images.
    /// @param leftImage
    /// The left input image as a `cv::Mat`.
    /// @param rightImage
    /// The right input image as a `cv::Mat`.
    /// @param disparity
    /// The output disparity map as a `cv::Mat`.
    /// @param maxDisparityDifference
    /// Maximum allowed disparity difference for left-right consistency check.
    /// If the value is less than or equal to 0, the check will be skipped.
    /// @return
    /// A Status object indicating whether the operation succeeded.
    [[nodiscard]] auto RunWithLeftRightConsistencyCheck(const cv::Mat &leftImage, const cv::Mat &rightImage, cv::Mat &disparity, //
                                                        float maxDisparityDifference = 1.0F) const noexcept -> Status;

  private:
    [[nodiscard]] auto RunImpl(const cv::Mat &leftImage, const cv::Mat &rightImage, cv::Mat &disparity, //
                               float maxDisparityDifference) const noexcept -> Status;

    size_t matchingHeight_{0};
    size_t matchingWidth_{0};
    Pipeline pipeline_;
};
} // namespace retinify::tools