// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/tools/pipeline.hpp"
#include "retinify/log.hpp"

#include <opencv2/imgproc.hpp>

namespace retinify::tools
{
inline static auto LRConsistencyCheck(const cv::Mat &leftDisparity, const cv::Mat &rightDisparity, cv::Mat &disparity, float maxDisparityDifference) -> bool
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
                    int right_x = x - static_cast<int>(left_d + 0.5F);

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

auto LRConsistencyPipeline::Initialize(PipelineResolution resolution) noexcept -> Status
{
    switch (resolution)
    {
    case PipelineResolution::LOW:
        imageHeight_ = 320;
        imageWidth_ = 640;
        break;
    case PipelineResolution::MEDIUM:
        imageHeight_ = 480;
        imageWidth_ = 640;
        break;
    case PipelineResolution::HIGH:
        imageHeight_ = 720;
        imageWidth_ = 1280;
        break;
    default:
        LogError("Invalid resolution specified for LRConsistencyPipeline.");
        return Status{StatusCategory::USER, StatusCode::INVALID_ARGUMENT};
    }

    auto status = pipeline_.Initialize(imageHeight_, imageWidth_);
    return status;
}

auto LRConsistencyPipeline::Run(const cv::Mat &leftImage, const cv::Mat &rightImage, cv::Mat &disparity, const float maxDisparityDifference) const noexcept -> Status
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

        // Check if both images have the same number of channels
        if (leftImage.channels() != rightImage.channels())
        {
            LogError("Left and right images have different number of channels.");
            return Status{StatusCategory::USER, StatusCode::INVALID_ARGUMENT};
        }

        // Only support 1 (grayscale) or 3 (RGB) channel images
        if (leftImage.channels() != 1 && leftImage.channels() != 3)
        {
            LogError("Only 1 (grayscale) or 3 (RGB) channel images are supported.");
            return Status{StatusCategory::USER, StatusCode::INVALID_ARGUMENT};
        }

        cv::Mat leftGray;
        cv::Mat rightGray;
        cv::Mat leftGrayFlipped;
        cv::Mat rightGrayFlipped;

        // Convert to grayscale if needed
        if (leftImage.channels() == 3)
        {
            cv::cvtColor(leftImage, leftGray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(rightImage, rightGray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            leftGray = leftImage.clone();
            rightGray = rightImage.clone();
        }

        cv::resize(leftGray, leftGray, cv::Size(imageWidth_, imageHeight_));
        cv::resize(rightGray, rightGray, cv::Size(imageWidth_, imageHeight_));
        leftGray.convertTo(leftGray, CV_32FC1);
        rightGray.convertTo(rightGray, CV_32FC1);
        cv::Mat leftDisparity = cv::Mat::zeros(leftGray.size(), CV_32FC1);

        auto leftStatus = pipeline_.Run(leftGray.ptr(), leftGray.step, rightGray.ptr(), rightGray.step, leftDisparity.ptr(), leftDisparity.step);
        if (!leftStatus.IsOK())
        {
            return leftStatus;
        }

        cv::flip(leftGray, leftGrayFlipped, 1);
        cv::flip(rightGray, rightGrayFlipped, 1);
        cv::Mat rightDisparity = cv::Mat::zeros(rightGray.size(), CV_32FC1);

        auto rightStatus = pipeline_.Run(rightGrayFlipped.ptr(), rightGrayFlipped.step, leftGrayFlipped.ptr(), leftGrayFlipped.step, rightDisparity.ptr(), rightDisparity.step);
        if (!rightStatus.IsOK())
        {
            return rightStatus;
        }

        cv::flip(rightDisparity, rightDisparity, 1);

        // LRCheck
        cv::Mat lrCheckedDisparity = cv::Mat::zeros(leftDisparity.size(), CV_32FC1);
        if (!LRConsistencyCheck(leftDisparity, rightDisparity, lrCheckedDisparity, maxDisparityDifference))
        {
            return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
        }

        // resize disparity to original size
        cv::resize(lrCheckedDisparity, disparity, leftImage.size(), 0, 0, cv::INTER_NEAREST);
        disparity = disparity * (static_cast<float>(leftImage.cols) / static_cast<float>(imageWidth_));
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