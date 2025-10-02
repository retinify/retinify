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
        (void)left8U_.Free();
        (void)right8U_.Free();
        (void)leftDisparity32FC1_.Free();
        (void)leftDisparityFiltered32FC1_.Free();
        (void)leftResized8U_.Free();
        (void)rightResized8U_.Free();
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

    auto Initialize(const std::uint32_t imageWidth, const std::uint32_t imageHeight, const PixelFormat pixelFormat, //
                    const DepthMode depthMode) noexcept -> Status
    {
        Status status;

        if ((imageWidth <= 0) || (imageHeight <= 0))
        {
            LogError("Image height and width must be greater than zero.");
            status = Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
            return status;
        }

        // Set image dimensions
        imageWidth_ = static_cast<std::size_t>(imageWidth);
        imageHeight_ = static_cast<std::size_t>(imageHeight);

        // Set image channels
        switch (pixelFormat)
        {
        case PixelFormat::GRAY8:
            imageChannels_ = 1;
            break;
        case PixelFormat::RGB8:
            imageChannels_ = 3;
            break;
        default:
            LogError("Invalid pixel format.");
            status = Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
            return status;
        }

        // Set matching resolution
        switch (depthMode)
        {
        case DepthMode::FAST:
            matchingWidth_ = 640;
            matchingHeight_ = 320;
            break;
        case DepthMode::BALANCED:
            matchingWidth_ = 640;
            matchingHeight_ = 480;
            break;
        case DepthMode::ACCURATE:
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

        status = left8U_.Allocate(imageHeight_, imageWidth_, imageChannels_, sizeof(std::uint8_t), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = right8U_.Allocate(imageHeight_, imageWidth_, imageChannels_, sizeof(std::uint8_t), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = leftDisparity32FC1_.Allocate(imageHeight_, imageWidth_, 1, sizeof(float), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = leftDisparityFiltered32FC1_.Allocate(imageHeight_, imageWidth_, 1, sizeof(float), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = leftResized8U_.Allocate(matchingHeight_, matchingWidth_, imageChannels_, sizeof(std::uint8_t), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = rightResized8U_.Allocate(matchingHeight_, matchingWidth_, imageChannels_, sizeof(std::uint8_t), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = leftResized8UC1_.Allocate(matchingHeight_, matchingWidth_, 1, sizeof(std::uint8_t), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = rightResized8UC1_.Allocate(matchingHeight_, matchingWidth_, 1, sizeof(std::uint8_t), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = leftResized32FC1_.Allocate(matchingHeight_, matchingWidth_, 1, sizeof(float), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = rightResized32FC1_.Allocate(matchingHeight_, matchingWidth_, 1, sizeof(float), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = disparityResized32FC1_.Allocate(matchingHeight_, matchingWidth_, 1, sizeof(float), MatLocation::DEVICE);
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

        if (leftImageStride < imageWidth_ * imageChannels_ * sizeof(std::uint8_t))
        {
            LogError("Left image stride is too small.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        if (rightImageStride < imageWidth_ * imageChannels_ * sizeof(std::uint8_t))
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
             float *disparityData, std::size_t disparityStride) noexcept -> Status
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

        status = left8U_.Upload(leftImageData, leftImageStride, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = right8U_.Upload(rightImageData, rightImageStride, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = ResizeImage8U(left8U_, leftResized8U_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = ResizeImage8U(right8U_, rightResized8U_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = ConvertImage8UToC1(leftResized8U_, leftResized8UC1_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = ConvertImage8UToC1(rightResized8U_, rightResized8UC1_, stream_);
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

        status = session_.Run(stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = ResizeDisparity32FC1(disparityResized32FC1_, leftDisparity32FC1_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = DisparityOcclusionFilter32FC1(leftDisparity32FC1_, leftDisparityFiltered32FC1_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = leftDisparityFiltered32FC1_.Download(disparityData, disparityStride, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = stream_.Synchronize();
        if (!status.IsOK())
        {
            return status;
        }

        return status;
    }

  private:
    bool initialized_{false};        // whether the pipeline is initialized
    Stream stream_;                  // stream for operations
    std::size_t imageWidth_{};       // original input image width
    std::size_t imageHeight_{};      // original input image height
    std::size_t imageChannels_{};    // original input image channels
    std::size_t matchingHeight_{};   // image height for stereo matching
    std::size_t matchingWidth_{};    // image width for stereo matching
    Session session_;                // inference session
    Mat left8U_;                     // input left image
    Mat right8U_;                    // input right image
    Mat leftDisparity32FC1_;         // output left disparity map
    Mat leftDisparityFiltered32FC1_; // output left disparity map after occlusion filtering
    Mat leftResized8U_;              // resized left image
    Mat rightResized8U_;             // resized right image
    Mat leftResized8UC1_;            // resized left gray image
    Mat rightResized8UC1_;           // resized right gray image
    Mat leftResized32FC1_;           // resized gray image for stereo matching
    Mat rightResized32FC1_;          // resized gray image for stereo matching
    Mat disparityResized32FC1_;      // resized disparity map from stereo matching
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

auto Pipeline::Initialize(const std::uint32_t imageWidth, const std::uint32_t imageHeight, PixelFormat pixelFormat, DepthMode depthMode) noexcept -> Status
{
    return this->impl()->Initialize(imageWidth, imageHeight, pixelFormat, depthMode);
}

auto Pipeline::Run(const std::uint8_t *leftImageData, std::size_t leftImageStride,   //
                   const std::uint8_t *rightImageData, std::size_t rightImageStride, //
                   float *disparityData, std::size_t disparityStride) noexcept -> Status
{
    return this->impl()->Run(leftImageData, leftImageStride, rightImageData, rightImageStride, disparityData, disparityStride);
}
} // namespace retinify
