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
    /// Initializes the stereo matching pipeline with the specified processing mode.
    /// @param mode
    /// The processing mode to use for the stereo matching pipeline.
    /// @return
    /// A Status object indicating whether initialization succeeded.
    [[nodiscard]] auto Initialize(int imageHeight, int imageWidth, Mode mode = Mode::ACCURATE) noexcept -> Status;

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
    /// @note
    /// Input images are resized internally, so as long as the left and right
    /// images share the same dimensions, their original sizes don’t matter.
    [[nodiscard]] auto Run(const cv::Mat &leftImage, const cv::Mat &rightImage, cv::Mat &disparity) noexcept -> Status;

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
    /// @note
    /// Input images are resized internally, so as long as the left and right
    /// images share the same dimensions, their original sizes don’t matter.
    [[nodiscard]] auto RunWithLeftRightConsistencyCheck(const cv::Mat &leftImage, const cv::Mat &rightImage, cv::Mat &disparity, //
                                                        float maxDisparityDifference = 1.0F) noexcept -> Status;

  private:
    [[nodiscard]] auto RunImpl(const cv::Mat &leftImage, const cv::Mat &rightImage, cv::Mat &disparity, //
                               float maxDisparityDifference) noexcept -> Status;

    size_t imageHeight_{0};
    size_t imageWidth_{0};
    Pipeline pipeline_;
};
} // namespace retinify::tools