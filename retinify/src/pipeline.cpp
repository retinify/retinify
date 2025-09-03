// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "imgproc.hpp"
#include "mat.hpp"
#include "session.hpp"

#include "retinify/log.hpp"
#include "retinify/path.hpp"
#include "retinify/pipeline.hpp"
#include "retinify/status.hpp"

#include <new>

namespace retinify
{
class Pipeline::Impl
{
  public:
    Impl() noexcept = default;

    ~Impl() noexcept
    {
        initialized_ = false;
        (void)left8UC3_.Free();
        (void)right8UC3_.Free();
        (void)disparity32FC1_.Free();
        (void)leftResized8UC3_.Free();
        (void)rightResized8UC3_.Free();
        (void)leftResized8UC1_.Free();
        (void)rightResized8UC1_.Free();
        (void)leftResized32FC1_.Free();
        (void)rightResized32FC1_.Free();
        (void)disparityResized32FC1_.Free();
    }

    Impl(const Impl &) = delete;
    auto operator=(const Impl &) noexcept -> Impl & = delete;
    Impl(Impl &&) noexcept = delete;
    auto operator=(Impl &&other) noexcept -> Impl & = delete;

    auto Initialize(const std::size_t imageHeight, const std::size_t imageWidth, const Mode mode) noexcept -> Status
    {
        Status status;

        if ((imageHeight <= 0) || (imageWidth <= 0))
        {
            LogError("Image height and width must be greater than zero.");
            status = Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
            return status;
        }

        imageHeight_ = imageHeight;
        imageWidth_ = imageWidth;

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
            LogError("Invalid stereo matching mode.");
            status = Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
            return status;
        }

        status = left8UC3_.Allocate(imageHeight_, imageWidth_, 3, sizeof(std::uint8_t));
        if (!status.IsOK())
        {
            return status;
        }

        status = right8UC3_.Allocate(imageHeight_, imageWidth_, 3, sizeof(std::uint8_t));
        if (!status.IsOK())
        {
            return status;
        }

        status = disparity32FC1_.Allocate(imageHeight_, imageWidth_, 1, sizeof(float));
        if (!status.IsOK())
        {
            return status;
        }

        status = leftResized8UC3_.Allocate(matchingHeight_, matchingWidth_, 3, sizeof(std::uint8_t));
        if (!status.IsOK())
        {
            return status;
        }

        status = rightResized8UC3_.Allocate(matchingHeight_, matchingWidth_, 3, sizeof(std::uint8_t));
        if (!status.IsOK())
        {
            return status;
        }

        status = leftResized8UC1_.Allocate(matchingHeight_, matchingWidth_, 1, sizeof(std::uint8_t));
        if (!status.IsOK())
        {
            return status;
        }

        status = rightResized8UC1_.Allocate(matchingHeight_, matchingWidth_, 1, sizeof(std::uint8_t));
        if (!status.IsOK())
        {
            return status;
        }

        status = leftResized32FC1_.Allocate(matchingHeight_, matchingWidth_, 1, sizeof(float));
        if (!status.IsOK())
        {
            return status;
        }

        status = rightResized32FC1_.Allocate(matchingHeight_, matchingWidth_, 1, sizeof(float));
        if (!status.IsOK())
        {
            return status;
        }

        status = disparityResized32FC1_.Allocate(matchingHeight_, matchingWidth_, 1, sizeof(float));
        if (!status.IsOK())
        {
            return status;
        }

        status = session_.Initialize(ONNXModelFilePath());
        if (!status.IsOK())
        {
            return status;
        }

        status = session_.BindInput("left", leftResized32FC1_);
        if (!status.IsOK())
        {
            return status;
        }

        status = session_.BindInput("right", rightResized32FC1_);
        if (!status.IsOK())
        {
            return status;
        }

        status = session_.BindOutput("disparity", disparityResized32FC1_);
        if (!status.IsOK())
        {
            return status;
        }

        initialized_ = true;
        return status;
    }

    [[nodiscard]] auto CheckInputImage(const std::uint8_t *leftImageData, const std::size_t leftImageStride, const std::uint8_t *rightImageData, const std::size_t rightImageStride, float *disparityData, const std::size_t disparityStride) noexcept -> Status
    {
        if (!leftImageData)
        {
            LogError("Left image data is nullptr.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        if (!rightImageData)
        {
            LogError("Right image data is nullptr.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        if (!disparityData)
        {
            LogError("Output disparity data is nullptr.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        if (leftImageStride < matchingWidth_ * sizeof(std::uint8_t))
        {
            LogError("Left image stride is too small.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        if (rightImageStride < matchingWidth_ * sizeof(std::uint8_t))
        {
            LogError("Right image stride is too small.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        if (disparityStride < matchingWidth_ * sizeof(float))
        {
            LogError("Disparity stride is too small.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        return Status{};
    }

    auto Run(const std::uint8_t *leftImageData, const std::size_t leftImageStride, const std::uint8_t *rightImageData, const std::size_t rightImageStride, float *disparityData, const std::size_t disparityStride) noexcept -> Status
    {
        Status status;

        if (!initialized_)
        {
            LogError("Pipeline is not initialized. Call Initialize() before Run().");
            status = Status(StatusCategory::USER, StatusCode::FAIL);
            return status;
        }

        status = CheckInputImage(leftImageData, leftImageStride, rightImageData, rightImageStride, disparityData, disparityStride);
        if (!status.IsOK())
        {
            return status;
        }

        status = left8UC3_.Upload(leftImageData, leftImageStride);
        if (!status.IsOK())
        {
            return status;
        }

        status = right8UC3_.Upload(rightImageData, rightImageStride);
        if (!status.IsOK())
        {
            return status;
        }

        status = ResizeImage8UC3(left8UC3_, leftResized8UC3_);
        if (!status.IsOK())
        {
            return status;
        }

        status = ResizeImage8UC3(right8UC3_, rightResized8UC3_);
        if (!status.IsOK())
        {
            return status;
        }

        status = Convert8UC3To8UC1(leftResized8UC3_, leftResized8UC1_);
        if (!status.IsOK())
        {
            return status;
        }

        status = Convert8UC3To8UC1(rightResized8UC3_, rightResized8UC1_);
        if (!status.IsOK())
        {
            return status;
        }

        status = Convert8UC1To32FC1(leftResized8UC1_, leftResized32FC1_);
        if (!status.IsOK())
        {
            return status;
        }

        status = Convert8UC1To32FC1(rightResized8UC1_, rightResized32FC1_);
        if (!status.IsOK())
        {
            return status;
        }

        status = leftResized32FC1_.Wait();
        if (!status.IsOK())
        {
            return status;
        }

        status = rightResized32FC1_.Wait();
        if (!status.IsOK())
        {
            return status;
        }

        status = disparityResized32FC1_.Wait();
        if (!status.IsOK())
        {
            return status;
        }

        status = session_.Run();
        if (!status.IsOK())
        {
            return status;
        }

        status = ResizeDisparity32FC1(disparityResized32FC1_, disparity32FC1_);
        if (!status.IsOK())
        {
            return status;
        }

        status = disparity32FC1_.Download(disparityData, disparityStride);
        if (!status.IsOK())
        {
            return status;
        }

        status = disparity32FC1_.Wait();
        if (!status.IsOK())
        {
            return status;
        }

        return status;
    }

  private:
    bool initialized_{false};

    size_t imageHeight_{0};     // original input image height
    size_t imageWidth_{0};      // original input image width
    size_t matchingHeight_{0};  // image height for stereo matching
    size_t matchingWidth_{0};   // image width for stereo matching
    Session session_;           // inference session
    Mat left8UC3_;              // input rgb image
    Mat right8UC3_;             // input rgb image
    Mat disparity32FC1_;        // output disparity map
    Mat leftResized8UC3_;       // resized rgb image
    Mat rightResized8UC3_;      // resized rgb image
    Mat leftResized8UC1_;       // resized gray image
    Mat rightResized8UC1_;      // resized gray image
    Mat leftResized32FC1_;      // resized gray image for stereo matching
    Mat rightResized32FC1_;     // resized gray image for stereo matching
    Mat disparityResized32FC1_; // resized output disparity map from stereo matching
};

Pipeline::Pipeline() noexcept
{
    static_assert(sizeof(buffer_) >= sizeof(Impl), "Buffer too small for Impl");
    static_assert(alignof(std::max_align_t) >= alignof(Impl), "Buffer alignment insufficient");

    new (&buffer_) Impl();
}

Pipeline::~Pipeline() noexcept
{
    impl()->~Impl();
}

auto Pipeline::impl() noexcept -> Impl *
{
    return std::launder(reinterpret_cast<Impl *>(&buffer_)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

auto Pipeline::impl() const noexcept -> const Impl *
{
    return std::launder(reinterpret_cast<const Impl *>(&buffer_)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

auto Pipeline::Initialize(std::size_t imageHeight, std::size_t imageWidth, Mode mode) noexcept -> Status
{
    return this->impl()->Initialize(imageHeight, imageWidth, mode);
}

auto Pipeline::Run(const std::uint8_t *leftImageData, std::size_t leftImageStride, const std::uint8_t *rightImageData, std::size_t rightImageStride, float *disparityData, std::size_t disparityStride) noexcept -> Status
{
    return this->impl()->Run(leftImageData, leftImageStride, rightImageData, rightImageStride, disparityData, disparityStride);
}
} // namespace retinify