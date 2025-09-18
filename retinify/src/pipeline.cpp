// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "imgproc.hpp"
#include "mat.hpp"
#include "session.hpp"
#include "stream.hpp"

#include "retinify/logging.hpp"
#include "retinify/paths.hpp"
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
        (void)stream_.Destroy();
        (void)left8UC3_.Free();
        (void)right8UC3_.Free();
        (void)leftDisparity32FC1_.Free();
        (void)leftResized8UC3_.Free();
        (void)rightResized8UC3_.Free();
        (void)leftResized8UC1_.Free();
        (void)rightResized8UC1_.Free();
        (void)leftResized32FC1_.Free();
        (void)rightResized32FC1_.Free();
        (void)disparityResized32FC1_.Free();
        (void)leftFliped8UC3_.Free();
        (void)rightFliped8UC3_.Free();
        (void)disparityFliped32FC1_.Free();
        (void)rightDisparity32FC1_.Free();
        (void)lrCheckedDisparity32FC1_.Free();
    }

    Impl(const Impl &) = delete;
    auto operator=(const Impl &) noexcept -> Impl & = delete;
    Impl(Impl &&) noexcept = delete;
    auto operator=(Impl &&other) noexcept -> Impl & = delete;

    auto Initialize(const std::size_t imageWidth, const std::size_t imageHeight, //
                    const Mode mode) noexcept -> Status
    {
        Status status;

        if ((imageWidth <= 0) || (imageHeight <= 0))
        {
            LogError("Image height and width must be greater than zero.");
            status = Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
            return status;
        }

        imageWidth_ = imageWidth;
        imageHeight_ = imageHeight;

        switch (mode)
        {
        case Mode::FAST:
            matchingWidth_ = 640;
            matchingHeight_ = 320;
            break;
        case Mode::BALANCED:
            matchingWidth_ = 640;
            matchingHeight_ = 480;
            break;
        case Mode::ACCURATE:
            matchingWidth_ = 1280;
            matchingHeight_ = 720;
            break;
        default:
            LogError("Invalid stereo matching mode.");
            status = Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
            return status;
        }

        status = stream_.Create();
        if (!status.IsOK())
        {
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

        status = leftDisparity32FC1_.Allocate(imageHeight_, imageWidth_, 1, sizeof(float));
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

        status = leftFliped8UC3_.Allocate(imageHeight_, imageWidth_, 3, sizeof(std::uint8_t));
        if (!status.IsOK())
        {
            return status;
        }

        status = rightFliped8UC3_.Allocate(imageHeight_, imageWidth_, 3, sizeof(std::uint8_t));
        if (!status.IsOK())
        {
            return status;
        }

        status = disparityFliped32FC1_.Allocate(imageHeight_, imageWidth_, 1, sizeof(float));
        if (!status.IsOK())
        {
            return status;
        }

        status = rightDisparity32FC1_.Allocate(imageHeight_, imageWidth_, 1, sizeof(float));
        if (!status.IsOK())
        {
            return status;
        }

        status = lrCheckedDisparity32FC1_.Allocate(imageHeight_, imageWidth_, 1, sizeof(float));
        if (!status.IsOK())
        {
            return status;
        }

        initialized_ = true;
        return status;
    }

    [[nodiscard]] auto CheckInputImage(const std::uint8_t *leftImageData, const std::size_t leftImageStride,   //
                                       const std::uint8_t *rightImageData, const std::size_t rightImageStride, //
                                       float *disparityData, const std::size_t disparityStride) noexcept -> Status
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

        if (leftImageStride < imageWidth_ * 3 * sizeof(std::uint8_t))
        {
            LogError("Left image stride is too small.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        if (rightImageStride < imageWidth_ * 3 * sizeof(std::uint8_t))
        {
            LogError("Right image stride is too small.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        if (disparityStride < imageWidth_ * sizeof(float))
        {
            LogError("Disparity stride is too small.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        return Status{};
    }

    auto Run(const std::uint8_t *leftImageData, std::size_t leftImageStride,   //
             const std::uint8_t *rightImageData, std::size_t rightImageStride, //
             float *disparityData, std::size_t disparityStride,                //
             float maxRelativeDisparityError) noexcept -> Status
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

        status = left8UC3_.Upload(leftImageData, leftImageStride, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = right8UC3_.Upload(rightImageData, rightImageStride, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = ResizeImage8UC3(left8UC3_, leftResized8UC3_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = ResizeImage8UC3(right8UC3_, rightResized8UC3_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = Convert8UC3To8UC1(leftResized8UC3_, leftResized8UC1_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = Convert8UC3To8UC1(rightResized8UC3_, rightResized8UC1_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = Convert8UC1To32FC1(leftResized8UC1_, leftResized32FC1_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = Convert8UC1To32FC1(rightResized8UC1_, rightResized32FC1_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        // Ensure all pre-processing has completed before inference
        status = stream_.Synchronize();
        if (!status.IsOK())
        {
            return status;
        }

        status = session_.Run();
        if (!status.IsOK())
        {
            return status;
        }

        status = ResizeDisparity32FC1(disparityResized32FC1_, leftDisparity32FC1_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        // Left-right consistency check
        if (maxRelativeDisparityError > 0.0f && maxRelativeDisparityError < 1.0f)
        {
            status = HorizontalFlip8UC3(left8UC3_, leftFliped8UC3_, stream_);
            if (!status.IsOK())
            {
                return status;
            }

            status = HorizontalFlip8UC3(right8UC3_, rightFliped8UC3_, stream_);
            if (!status.IsOK())
            {
                return status;
            }

            status = ResizeImage8UC3(leftFliped8UC3_, leftResized8UC3_, stream_);
            if (!status.IsOK())
            {
                return status;
            }

            status = ResizeImage8UC3(rightFliped8UC3_, rightResized8UC3_, stream_);
            if (!status.IsOK())
            {
                return status;
            }

            status = Convert8UC3To8UC1(leftResized8UC3_, leftResized8UC1_, stream_);
            if (!status.IsOK())
            {
                return status;
            }

            status = Convert8UC3To8UC1(rightResized8UC3_, rightResized8UC1_, stream_);
            if (!status.IsOK())
            {
                return status;
            }

            status = Convert8UC1To32FC1(rightResized8UC1_, leftResized32FC1_, stream_);
            if (!status.IsOK())
            {
                return status;
            }

            status = Convert8UC1To32FC1(leftResized8UC1_, rightResized32FC1_, stream_);
            if (!status.IsOK())
            {
                return status;
            }

            // Ensure all pre-processing has completed before inference
            status = stream_.Synchronize();
            if (!status.IsOK())
            {
                return status;
            }

            status = session_.Run();
            if (!status.IsOK())
            {
                return status;
            }

            status = ResizeDisparity32FC1(disparityResized32FC1_, disparityFliped32FC1_, stream_);
            if (!status.IsOK())
            {
                return status;
            }

            status = HorizontalFlip32FC1(disparityFliped32FC1_, rightDisparity32FC1_, stream_);
            if (!status.IsOK())
            {
                return status;
            }

            status = LRConsistencyCheck32FC1(leftDisparity32FC1_, rightDisparity32FC1_, lrCheckedDisparity32FC1_, maxRelativeDisparityError, stream_);
            if (!status.IsOK())
            {
                return status;
            }

            status = lrCheckedDisparity32FC1_.Download(disparityData, disparityStride, stream_);
            if (!status.IsOK())
            {
                return status;
            }
        }
        else
        {
            status = leftDisparity32FC1_.Download(disparityData, disparityStride, stream_);
            if (!status.IsOK())
            {
                return status;
            }
        }

        status = stream_.Synchronize();
        if (!status.IsOK())
        {
            return status;
        }

        return status;
    }

  private:
    bool initialized_{false};     // whether the pipeline is initialized
    Stream stream_;               // stream for operations
    size_t imageWidth_{0};        // original input image width
    size_t imageHeight_{0};       // original input image height
    size_t matchingHeight_{0};    // image height for stereo matching
    size_t matchingWidth_{0};     // image width for stereo matching
    Session session_;             // inference session
    Mat left8UC3_;                // input rgb image
    Mat right8UC3_;               // input rgb image
    Mat leftDisparity32FC1_;      // output left disparity map
    Mat leftResized8UC3_;         // resized rgb image
    Mat rightResized8UC3_;        // resized rgb image
    Mat leftResized8UC1_;         // resized gray image
    Mat rightResized8UC1_;        // resized gray image
    Mat leftResized32FC1_;        // resized gray image for stereo matching
    Mat rightResized32FC1_;       // resized gray image for stereo matching
    Mat disparityResized32FC1_;   // resized disparity map from stereo matching
    Mat leftFliped8UC3_;          // input gray image (fliped)
    Mat rightFliped8UC3_;         // input gray image (fliped)
    Mat disparityFliped32FC1_;    // output disparity map (fliped)
    Mat rightDisparity32FC1_;     // output right disparity map
    Mat lrCheckedDisparity32FC1_; // output disparity map after left-right consistency check
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

auto Pipeline::Initialize(std::size_t imageWidth, std::size_t imageHeight, Mode mode) noexcept -> Status
{
    return this->impl()->Initialize(imageWidth, imageHeight, mode);
}

auto Pipeline::Run(const std::uint8_t *leftImageData, std::size_t leftImageStride,   //
                   const std::uint8_t *rightImageData, std::size_t rightImageStride, //
                   float *disparityData, std::size_t disparityStride) noexcept -> Status
{
    constexpr float kMaxDisparityDifference = -1.0f; // Disable left-right consistency check
    return this->impl()->Run(leftImageData, leftImageStride, rightImageData, rightImageStride, disparityData, disparityStride, kMaxDisparityDifference);
}

auto Pipeline::Run(const std::uint8_t *leftImageData, std::size_t leftImageStride,   //
                   const std::uint8_t *rightImageData, std::size_t rightImageStride, //
                   float *disparityData, std::size_t disparityStride,                //
                   const float maxRelativeDisparityError) noexcept -> Status
{
    if (maxRelativeDisparityError <= 0.0f || maxRelativeDisparityError >= 1.0f)
    {
        LogWarn("maxRelativeDisparityError should be in the range (0.0, 1.0). Left-right consistency check is disabled.");
    }
    return this->impl()->Run(leftImageData, leftImageStride, rightImageData, rightImageStride, disparityData, disparityStride, maxRelativeDisparityError);
}
} // namespace retinify
