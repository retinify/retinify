// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-eula

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
        (void)leftMapX_.Free();
        (void)leftMapY_.Free();
        (void)rightMapX_.Free();
        (void)rightMapY_.Free();
        (void)left8U_.Free();
        (void)right8U_.Free();
        (void)leftRectified8U_.Free();
        (void)rightRectified8U_.Free();
        (void)leftDisparity32FC1_.Free();
        (void)leftDisparityFiltered32FC1_.Free();
        (void)leftResizedRectified8U_.Free();
        (void)rightResizedRectified8U_.Free();
        (void)leftResizedRectified8UC1_.Free();
        (void)rightResizedRectified8UC1_.Free();
        (void)leftResizedRectified32FC1_.Free();
        (void)rightResizedRectified32FC1_.Free();
        (void)disparityResized32FC1_.Free();
        (void)pointCloud32FC3_.Free();
    }

    Impl(const Impl &) = delete;
    auto operator=(const Impl &) noexcept -> Impl & = delete;
    Impl(Impl &&) noexcept = delete;
    auto operator=(Impl &&other) noexcept -> Impl & = delete;

    auto Initialize(std::uint32_t imageWidth, std::uint32_t imageHeight, PixelFormat pixelFormat, //
                    DepthMode depthMode, const CalibrationParameters &calibrationParameters) noexcept -> Status
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

        // Initialize matrices
        reprojectionMatrix_ = Mat4x4d{};

        status = stream_.Create();
        if (!status.IsOK())
        {
            return status;
        }

        status = leftMapX_.Allocate(imageHeight_, imageWidth_, 1, sizeof(float), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = leftMapY_.Allocate(imageHeight_, imageWidth_, 1, sizeof(float), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = rightMapX_.Allocate(imageHeight_, imageWidth_, 1, sizeof(float), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = rightMapY_.Allocate(imageHeight_, imageWidth_, 1, sizeof(float), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        Mat leftMapXHost, leftMapYHost, rightMapXHost, rightMapYHost;
        status = leftMapXHost.Allocate(imageHeight_, imageWidth_, 1, sizeof(float), MatLocation::HOST);
        if (!status.IsOK())
        {
            return status;
        }

        status = leftMapYHost.Allocate(imageHeight_, imageWidth_, 1, sizeof(float), MatLocation::HOST);
        if (!status.IsOK())
        {
            return status;
        }

        status = rightMapXHost.Allocate(imageHeight_, imageWidth_, 1, sizeof(float), MatLocation::HOST);
        if (!status.IsOK())
        {
            return status;
        }

        status = rightMapYHost.Allocate(imageHeight_, imageWidth_, 1, sizeof(float), MatLocation::HOST);
        if (!status.IsOK())
        {
            return status;
        }

        if (calibrationParameters == CalibrationParameters{})
        {
            retinify::InitIdentityMap(static_cast<float *>(leftMapXHost.Data()), leftMapXHost.Stride(), //
                                      static_cast<float *>(leftMapYHost.Data()), leftMapYHost.Stride(), //
                                      imageWidth_, imageHeight_);

            retinify::InitIdentityMap(static_cast<float *>(rightMapXHost.Data()), rightMapXHost.Stride(), //
                                      static_cast<float *>(rightMapYHost.Data()), rightMapYHost.Stride(), //
                                      imageWidth_, imageHeight_);
        }
        else
        {
            retinify::Mat3x3d R1, R2;
            retinify::Mat3x4d P1, P2;

            retinify::StereoRectify(calibrationParameters.leftIntrinsics, calibrationParameters.leftDistortion,   //
                                    calibrationParameters.rightIntrinsics, calibrationParameters.rightDistortion, //
                                    calibrationParameters.rotation, calibrationParameters.translation,            //
                                    static_cast<std::uint32_t>(calibrationParameters.imageWidth),                 //
                                    static_cast<std::uint32_t>(calibrationParameters.imageHeight),                //
                                    R1, R2, P1, P2, reprojectionMatrix_, 0.0);

            retinify::InitUndistortRectifyMap(calibrationParameters.leftIntrinsics, calibrationParameters.leftDistortion, //
                                              R1, P1,                                                                     //
                                              static_cast<std::uint32_t>(calibrationParameters.imageWidth),               //
                                              static_cast<std::uint32_t>(calibrationParameters.imageHeight),              //
                                              static_cast<float *>(leftMapXHost.Data()), leftMapXHost.Stride(),           //
                                              static_cast<float *>(leftMapYHost.Data()), leftMapYHost.Stride());          //

            retinify::InitUndistortRectifyMap(calibrationParameters.rightIntrinsics, calibrationParameters.rightDistortion, //
                                              R2, P2,                                                                       //
                                              static_cast<std::uint32_t>(calibrationParameters.imageWidth),                 //
                                              static_cast<std::uint32_t>(calibrationParameters.imageHeight),                //
                                              static_cast<float *>(rightMapXHost.Data()), rightMapXHost.Stride(),           //
                                              static_cast<float *>(rightMapYHost.Data()), rightMapYHost.Stride());          //
        }

        status = leftMapX_.Upload(leftMapXHost.Data(), leftMapXHost.Stride(), stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = leftMapY_.Upload(leftMapYHost.Data(), leftMapYHost.Stride(), stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = rightMapX_.Upload(rightMapXHost.Data(), rightMapXHost.Stride(), stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = rightMapY_.Upload(rightMapYHost.Data(), rightMapYHost.Stride(), stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = leftMapXHost.Free();
        if (!status.IsOK())
        {
            return status;
        }

        status = leftMapYHost.Free();
        if (!status.IsOK())
        {
            return status;
        }

        status = rightMapXHost.Free();
        if (!status.IsOK())
        {
            return status;
        }

        status = rightMapYHost.Free();
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

        status = leftRectified8U_.Allocate(imageHeight_, imageWidth_, imageChannels_, sizeof(std::uint8_t), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = rightRectified8U_.Allocate(imageHeight_, imageWidth_, imageChannels_, sizeof(std::uint8_t), MatLocation::DEVICE);
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

        status = pointCloud32FC3_.Allocate(imageHeight_, imageWidth_, 3, sizeof(float), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = leftResizedRectified8U_.Allocate(matchingHeight_, matchingWidth_, imageChannels_, sizeof(std::uint8_t), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = rightResizedRectified8U_.Allocate(matchingHeight_, matchingWidth_, imageChannels_, sizeof(std::uint8_t), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = leftResizedRectified8UC1_.Allocate(matchingHeight_, matchingWidth_, 1, sizeof(std::uint8_t), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = rightResizedRectified8UC1_.Allocate(matchingHeight_, matchingWidth_, 1, sizeof(std::uint8_t), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = leftResizedRectified32FC1_.Allocate(matchingHeight_, matchingWidth_, 1, sizeof(float), MatLocation::DEVICE);
        if (!status.IsOK())
        {
            return status;
        }

        status = rightResizedRectified32FC1_.Allocate(matchingHeight_, matchingWidth_, 1, sizeof(float), MatLocation::DEVICE);
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

        status = session_.BindInput("left", leftResizedRectified32FC1_);
        if (!status.IsOK())
        {
            return status;
        }

        status = session_.BindInput("right", rightResizedRectified32FC1_);
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

        status = RemapImage8U(left8U_, leftMapX_, leftMapY_, leftRectified8U_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = RemapImage8U(right8U_, rightMapX_, rightMapY_, rightRectified8U_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = ResizeImage8U(leftRectified8U_, leftResizedRectified8U_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = ResizeImage8U(rightRectified8U_, rightResizedRectified8U_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = ConvertImage8UToC1(leftResizedRectified8U_, leftResizedRectified8UC1_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = ConvertImage8UToC1(rightResizedRectified8U_, rightResizedRectified8UC1_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = Convert8UC1To32FC1(leftResizedRectified8UC1_, leftResizedRectified32FC1_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = Convert8UC1To32FC1(rightResizedRectified8UC1_, rightResizedRectified32FC1_, stream_);
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

    [[nodiscard]] auto RetrieveRectifiedLeftImage(std::uint8_t *leftImageData, std::size_t leftImageStride) noexcept -> Status
    {
        Status status;

        if (!initialized_)
        {
            LogError("Pipeline is not initialized. Call Initialize() before RetrieveRectifiedLeftImage().");
            return Status(StatusCategory::USER, StatusCode::FAIL);
        }

        if (leftImageData == nullptr)
        {
            LogError("Output rectified left image data is nullptr.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        const std::size_t requiredStride = imageWidth_ * imageChannels_ * sizeof(std::uint8_t);
        if (leftImageStride < requiredStride)
        {
            LogError("Rectified left image stride is too small.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        status = leftRectified8U_.Download(leftImageData, leftImageStride, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = stream_.Synchronize();
        if (!status.IsOK())
        {
            return status;
        }

        return Status{};
    }

    [[nodiscard]] auto RetrieveRectifiedRightImage(std::uint8_t *rightImageData, std::size_t rightImageStride) noexcept -> Status
    {
        Status status;

        if (!initialized_)
        {
            LogError("Pipeline is not initialized. Call Initialize() before RetrieveRectifiedRightImage().");
            return Status(StatusCategory::USER, StatusCode::FAIL);
        }

        if (rightImageData == nullptr)
        {
            LogError("Output rectified right image data is nullptr.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        const std::size_t requiredStride = imageWidth_ * imageChannels_ * sizeof(std::uint8_t);
        if (rightImageStride < requiredStride)
        {
            LogError("Rectified right image stride is too small.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        status = rightRectified8U_.Download(rightImageData, rightImageStride, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = stream_.Synchronize();
        if (!status.IsOK())
        {
            return status;
        }

        return Status{};
    }

    [[nodiscard]] auto RetrieveRectifiedImages(std::uint8_t *leftImageData, std::size_t leftImageStride, //
                                               std::uint8_t *rightImageData, std::size_t rightImageStride) noexcept -> Status
    {
        Status status;

        if (!initialized_)
        {
            LogError("Pipeline is not initialized. Call Initialize() before RetrieveRectifiedImages().");
            return Status(StatusCategory::USER, StatusCode::FAIL);
        }

        if ((leftImageData == nullptr) || (rightImageData == nullptr))
        {
            LogError("Output rectified image data is nullptr.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        const std::size_t requiredStride = imageWidth_ * imageChannels_ * sizeof(std::uint8_t);
        if (leftImageStride < requiredStride)
        {
            LogError("Rectified left image stride is too small.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        if (rightImageStride < requiredStride)
        {
            LogError("Rectified right image stride is too small.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        status = leftRectified8U_.Download(leftImageData, leftImageStride, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = rightRectified8U_.Download(rightImageData, rightImageStride, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = stream_.Synchronize();
        if (!status.IsOK())
        {
            return status;
        }

        return Status{};
    }

    [[nodiscard]] auto RetrieveDisparity(float *disparityData, std::size_t disparityStride) noexcept -> Status
    {
        Status status;

        if (!initialized_)
        {
            LogError("Pipeline is not initialized. Call Initialize() before RetrieveDisparity().");
            return Status(StatusCategory::USER, StatusCode::FAIL);
        }

        if (disparityData == nullptr)
        {
            LogError("Output disparity data is nullptr.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        const std::size_t requiredStride = imageWidth_ * sizeof(float);
        if (disparityStride < requiredStride)
        {
            LogError("Disparity stride is too small.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
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

        return Status{};
    }

    [[nodiscard]] auto RetrievePointCloud(float *pointCloudData, std::size_t pointCloudStride) noexcept -> Status
    {
        Status status;

        if (!initialized_)
        {
            LogError("Pipeline is not initialized. Call Initialize() before ComputePointCloud().");
            return Status(StatusCategory::USER, StatusCode::FAIL);
        }

        if (pointCloudData == nullptr)
        {
            LogError("Output point cloud data is nullptr.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        const std::size_t requiredStride = imageWidth_ * 3 * sizeof(float);
        if (pointCloudStride < requiredStride)
        {
            LogError("Point cloud stride is too small.");
            return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
        }

        status = ReprojectDisparityTo3D(leftDisparityFiltered32FC1_, pointCloud32FC3_, reprojectionMatrix_, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = pointCloud32FC3_.Download(pointCloudData, pointCloudStride, stream_);
        if (!status.IsOK())
        {
            return status;
        }

        status = stream_.Synchronize();
        if (!status.IsOK())
        {
            return status;
        }

        return Status{};
    }

  private:
    bool initialized_{false};        // whether the pipeline is initialized
    std::size_t imageWidth_{};       // original input image width
    std::size_t imageHeight_{};      // original input image height
    std::size_t imageChannels_{};    // original input image channels
    std::size_t matchingHeight_{};   // image height for stereo matching
    std::size_t matchingWidth_{};    // image width for stereo matching
    Session session_;                // inference session
    Stream stream_;                  // stream for operations
    Mat leftMapX_;                   // left x map for image remapping
    Mat leftMapY_;                   // left y map for image remapping
    Mat rightMapX_;                  // right x map for image remapping
    Mat rightMapY_;                  // right y map for image remapping
    Mat left8U_;                     // input left image
    Mat right8U_;                    // input right image
    Mat leftRectified8U_;            // rectified left image
    Mat rightRectified8U_;           // rectified right image
    Mat leftDisparity32FC1_;         // output left disparity map
    Mat leftDisparityFiltered32FC1_; // output left disparity map after occlusion filtering
    Mat leftResizedRectified8U_;     // resized left image
    Mat rightResizedRectified8U_;    // resized right image
    Mat leftResizedRectified8UC1_;   // resized left gray image
    Mat rightResizedRectified8UC1_;  // resized right gray image
    Mat leftResizedRectified32FC1_;  // resized gray image for stereo matching
    Mat rightResizedRectified32FC1_; // resized gray image for stereo matching
    Mat disparityResized32FC1_;      // resized disparity map from stereo matching
    Mat pointCloud32FC3_;            // reprojected 3D point cloud
    Mat4x4d reprojectionMatrix_{};   // reprojection matrix (double)
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

auto Pipeline::Initialize(std::uint32_t imageWidth, std::uint32_t imageHeight, PixelFormat pixelFormat, //
                          DepthMode depthMode, const CalibrationParameters &calibrationParameters) noexcept -> Status
{
    return this->impl()->Initialize(imageWidth, imageHeight, pixelFormat, depthMode, calibrationParameters);
}

auto Pipeline::Run(const std::uint8_t *leftImageData, std::size_t leftImageStride,   //
                   const std::uint8_t *rightImageData, std::size_t rightImageStride, //
                   float *disparityData, std::size_t disparityStride) noexcept -> Status
{
    return this->impl()->Run(leftImageData, leftImageStride, rightImageData, rightImageStride, disparityData, disparityStride);
}

auto Pipeline::RetrieveRectifiedLeftImage(std::uint8_t *leftImageData, std::size_t leftImageStride) noexcept -> Status
{
    return this->impl()->RetrieveRectifiedLeftImage(leftImageData, leftImageStride);
}

auto Pipeline::RetrieveRectifiedRightImage(std::uint8_t *rightImageData, std::size_t rightImageStride) noexcept -> Status
{
    return this->impl()->RetrieveRectifiedRightImage(rightImageData, rightImageStride);
}

auto Pipeline::RetrieveRectifiedImages(std::uint8_t *leftImageData, std::size_t leftImageStride, //
                                       std::uint8_t *rightImageData, std::size_t rightImageStride) noexcept -> Status
{
    return this->impl()->RetrieveRectifiedImages(leftImageData, leftImageStride, rightImageData, rightImageStride);
}

auto Pipeline::RetrieveDisparity(float *disparityData, std::size_t disparityStride) noexcept -> Status
{
    return this->impl()->RetrieveDisparity(disparityData, disparityStride);
}

auto Pipeline::RetrievePointCloud(float *pointCloudData, std::size_t pointCloudStride) noexcept -> Status
{
    return this->impl()->RetrievePointCloud(pointCloudData, pointCloudStride);
}
} // namespace retinify
