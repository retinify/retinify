// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/tools/pipeline.hpp"
#include "retinify/log.hpp"

#include <opencv2/imgproc.hpp>

namespace retinify::tools
{
static inline auto LRConsistencyCheck(const cv::Mat &leftDisparity, const cv::Mat &rightDisparity, cv::Mat &disparity, float maxDisparityDifference) -> bool
{
    if (leftDisparity.empty() || rightDisparity.empty())
    {
        LogError("Left or right disparity map is empty.");
        return false;
    }

    if (leftDisparity.size() != rightDisparity.size())
    {
        LogError("Left and right disparity maps have different sizes.");
        return false;
    }

    if (leftDisparity.channels() != 1 || rightDisparity.channels() != 1)
    {
        LogError("Left and right disparity maps must be single channel.");
        return false;
    }

    disparity = cv::Mat::zeros(leftDisparity.size(), CV_32FC1);

    cv::parallel_for_(cv::Range(0, leftDisparity.rows), [leftDisparity, rightDisparity, &disparity, maxDisparityDifference](const cv::Range &range) {
        for (int y = range.start; y < range.end; y++)
        {
            const auto *leftPtr = leftDisparity.ptr<float>(y);
            const auto *rightPtr = rightDisparity.ptr<float>(y);
            auto *disparityPtr = disparity.ptr<float>(y);

            for (int x = 0; x < leftDisparity.cols; x++)
            {
                float left_d = leftPtr[x];
                if (left_d > 0)
                {
                    constexpr float kDisparityRoundOffset = 0.5F;
                    int right_x = x - static_cast<int>(left_d + kDisparityRoundOffset);

                    if (right_x >= 0 && right_x < rightDisparity.cols)
                    {
                        float right_d = rightPtr[right_x];

                        if (std::abs(left_d - right_d) <= maxDisparityDifference)
                        {
                            disparityPtr[x] = left_d;
                        }
                    }
                }
            }
        }
    });

    return true;
}

auto StereoMatchingPipeline::Initialize(Mode mode) noexcept -> Status
{
    switch (mode)
    {
    case Mode::FAST:
        matchingHeight_ = 320;
        matchingWidth_ = 640;
        break;
    case Mode::BALANCED:
        matchingHeight_ = 480;
        matchingWidth_ = 640;
        break;
    case Mode::ACCURATE:
        matchingHeight_ = 720;
        matchingWidth_ = 1280;
        break;
    default:
        LogError("Invalid mode specified for StereoMatchingPipeline.");
        return Status{StatusCategory::USER, StatusCode::INVALID_ARGUMENT};
    }

    auto status = pipeline_.Initialize(matchingHeight_, matchingWidth_);
    return status;
}

auto StereoMatchingPipeline::Run(const cv::Mat &leftImage, const cv::Mat &rightImage, cv::Mat &disparity) noexcept -> Status
{
    constexpr float kMaxDisparityDifference = -1.0F; // Disable left-right consistency check by default
    return RunImpl(leftImage, rightImage, disparity, kMaxDisparityDifference);
}

auto StereoMatchingPipeline::RunWithLeftRightConsistencyCheck(const cv::Mat &leftImage, const cv::Mat &rightImage, cv::Mat &disparity, const float maxDisparityDifference) noexcept -> Status
{
    if (maxDisparityDifference <= 0.0F)
    {
        LogWarn("Left-right consistency check is disabled due to non-positive maxDisparityDifference.");
    }
    return RunImpl(leftImage, rightImage, disparity, maxDisparityDifference);
}

auto StereoMatchingPipeline::RunImpl(const cv::Mat &leftImage, const cv::Mat &rightImage, cv::Mat &disparity, const float maxDisparityDifference) noexcept -> Status
{
    try
    {
        if (leftImage.empty() || rightImage.empty())
        {
            LogError("Left or right image is empty.");
            return Status{StatusCategory::USER, StatusCode::INVALID_ARGUMENT};
        }

        if (leftImage.size() != rightImage.size())
        {
            LogError("Left and right images have different sizes.");
            return Status{StatusCategory::USER, StatusCode::INVALID_ARGUMENT};
        }

        if (leftImage.channels() != rightImage.channels())
        {
            LogError("Left and right images have different number of channels.");
            return Status{StatusCategory::USER, StatusCode::INVALID_ARGUMENT};
        }

        if (leftImage.channels() != 1 && leftImage.channels() != 3)
        {
            LogError("Only 1 (grayscale) or 3 (RGB) channel images are supported.");
            return Status{StatusCategory::USER, StatusCode::INVALID_ARGUMENT};
        }

        cv::Mat leftRGB;
        cv::Mat rightRGB;
        leftRGB = leftImage.clone();
        rightRGB = rightImage.clone();

        cv::resize(leftRGB, leftRGB, cv::Size(matchingWidth_, matchingHeight_));
        cv::resize(rightRGB, rightRGB, cv::Size(matchingWidth_, matchingHeight_));
        cv::Mat leftDisparity = cv::Mat::zeros(leftRGB.size(), CV_32FC1);

        auto leftStatus = pipeline_.Run(leftRGB.ptr<std::uint8_t>(), leftRGB.step[0], rightRGB.ptr<std::uint8_t>(), rightRGB.step[0], leftDisparity.ptr<float>(), leftDisparity.step[0]);
        if (!leftStatus.IsOK())
        {
            return leftStatus;
        }

        // If maxDisparityDifference is greater than 0, perform left-right consistency check
        if (maxDisparityDifference > 0.0F)
        {
            // Scale maxDisparityDifference to fit the processing size
            float tmpMaxDisparityDifference = maxDisparityDifference * (static_cast<float>(matchingWidth_) / static_cast<float>(leftImage.cols));

            cv::Mat leftGrayFlipped;
            cv::Mat rightGrayFlipped;

            cv::flip(leftRGB, leftGrayFlipped, 1);
            cv::flip(rightRGB, rightGrayFlipped, 1);
            cv::Mat rightDisparity = cv::Mat::zeros(rightRGB.size(), CV_32FC1);

            auto rightStatus = pipeline_.Run(rightGrayFlipped.ptr<std::uint8_t>(), rightGrayFlipped.step[0], leftGrayFlipped.ptr<std::uint8_t>(), leftGrayFlipped.step[0], rightDisparity.ptr<float>(), rightDisparity.step[0]);
            if (!rightStatus.IsOK())
            {
                return rightStatus;
            }

            cv::flip(rightDisparity, rightDisparity, 1);

            cv::Mat lrCheckedDisparity = cv::Mat::zeros(leftDisparity.size(), CV_32FC1);
            if (!LRConsistencyCheck(leftDisparity, rightDisparity, lrCheckedDisparity, tmpMaxDisparityDifference))
            {
                return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
            }

            disparity = lrCheckedDisparity;
        }
        else
        {
            disparity = leftDisparity;
        }

        // resize disparity to original image size
        cv::resize(disparity, disparity, leftImage.size(), 0, 0, cv::INTER_NEAREST);
        disparity = disparity * (static_cast<float>(leftImage.cols) / static_cast<float>(matchingWidth_));
    }
    catch (const std::exception &e)
    {
        LogError(e.what());
        return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
    }
    catch (...)
    {
        LogError("Unknown error occurred in LRConsistencyPipeline::Run");
        return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
    }

    return Status{};
}
} // namespace retinify::tools