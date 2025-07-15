// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/tools/pipeline.hpp"
#include "retinify/log.hpp"

#include <opencv2/imgproc.hpp>

namespace retinify
{
namespace tools
{
inline static bool LRConsistencyCheck(const cv::Mat &leftDisparity, const cv::Mat &rightDisparity, cv::Mat &disparity, float threshold = 1.0f)
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

    disparity = cv::Mat::zeros(leftDisparity.size(), CV_32FC1);

    cv::parallel_for_(cv::Range(0, leftDisparity.rows), [leftDisparity, rightDisparity, &disparity, threshold](const cv::Range &range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float *leftPtr = leftDisparity.ptr<float>(y);
            const float *rightPtr = rightDisparity.ptr<float>(y);
            float *disparityPtr = disparity.ptr<float>(y);

            for (int x = 0; x < leftDisparity.cols; x++)
            {
                float left_d = leftPtr[x];
                if (left_d > 0)
                {
                    int right_x = x - static_cast<int>(left_d + 0.5f);

                    if (right_x >= 0 && right_x < rightDisparity.cols)
                    {
                        float right_d = rightPtr[right_x];

                        if (std::abs(left_d - right_d) <= threshold)
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

Status LRConsistencyPipeline::Initialize(std::size_t imageHeight, std::size_t imageWidth) noexcept
{
    return pipeline_.Initialize(imageHeight, imageWidth);
}

Status LRConsistencyPipeline::Run(const cv::Mat &leftImage, const cv::Mat &rightImage, cv::Mat &disparity) const noexcept
{
    try
    {

        if (leftImage.empty() || rightImage.empty())
        {
            return Status{StatusCategory::USER, StatusCode::INVALID_ARGUMENT};
        }

        if (leftImage.size() != rightImage.size())
        {
            return Status{StatusCategory::USER, StatusCode::INVALID_ARGUMENT};
        }

        cv::Mat leftGray, rightGray;
        cv::Mat leftGrayFlipped, rightGrayFlipped;

        cv::cvtColor(leftImage, leftGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rightImage, rightGray, cv::COLOR_BGR2GRAY);

        constexpr int width = 1280;
        constexpr int height = 720;
        cv::resize(leftGray, leftGray, cv::Size(width, height));
        cv::resize(rightGray, rightGray, cv::Size(width, height));
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
        if (!LRConsistencyCheck(leftDisparity, rightDisparity, lrCheckedDisparity))
        {
            return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
        }

        // resize disparity to original size
        cv::resize(lrCheckedDisparity, disparity, leftImage.size(), 0, 0, cv::INTER_NEAREST);
        disparity = disparity * (static_cast<float>(leftImage.cols) / width);
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
} // namespace tools
} // namespace retinify