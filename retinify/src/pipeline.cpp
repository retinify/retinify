// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mat.hpp"
#include "session.hpp"

#include "retinify/log.hpp"
#include "retinify/path.hpp"
#include "retinify/pipeline.hpp"
#include "retinify/status.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
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
        (void)left_.Free();
        (void)right_.Free();
        (void)disparity_.Free();
    }

    Status Initialize(const std::size_t height, const std::size_t width) noexcept
    {
        Status status;

        if (!((height == 320 && width == 640) || //
              (height == 480 && width == 640) || //
              (height == 720 && width == 1280)))
        {
            LogError("Height and width must be one of the following: 320x640, 480x640, or 720x1280.");
            status = Status(StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT);
            return status;
        }

        status = left_.Allocate(height, width, 3, sizeof(float));
        if (!status.IsOK())
        {
            return status;
        }

        status = right_.Allocate(height, width, 3, sizeof(float));
        if (!status.IsOK())
        {
            return status;
        }

        status = disparity_.Allocate(height, width, 1, sizeof(float));
        if (!status.IsOK())
        {
            return status;
        }

        status = session_.Initialize(ONNXModelFilePath());
        if (!status.IsOK())
        {
            return status;
        }

        status = session_.BindInput("left", left_);
        if (!status.IsOK())
        {
            return status;
        }

        status = session_.BindInput("right", right_);
        if (!status.IsOK())
        {
            return status;
        }

        status = session_.BindOutput("disparity", disparity_);
        if (!status.IsOK())
        {
            return status;
        }

        initialized_ = true;
        return status;
    }

    Status Run(const void *leftData, const std::size_t leftStride, const void *rightData, const std::size_t rightStride, void *disparityData, const std::size_t disparityStride) const noexcept
    {
        Status status;

        if (!initialized_)
        {
            status = Status(StatusCategory::RETINIFY, StatusCode::FAIL);
            return status;
        }

        status = left_.Upload(leftData, leftStride);
        if (!status.IsOK())
        {
            return status;
        }

        status = right_.Upload(rightData, rightStride);
        if (!status.IsOK())
        {
            return status;
        }

        status = left_.Wait();
        if (!status.IsOK())
        {
            return status;
        }

        status = right_.Wait();
        if (!status.IsOK())
        {
            return status;
        }

        status = disparity_.Wait();
        if (!status.IsOK())
        {
            return status;
        }

        status = session_.Run();
        if (!status.IsOK())
        {
            return status;
        }

        status = disparity_.Download(disparityData, disparityStride);
        if (!status.IsOK())
        {
            return status;
        }

        status = disparity_.Wait();
        if (!status.IsOK())
        {
            return status;
        }

        return status;
    }

  private:
    bool initialized_{false};
    Session session_;
    Mat left_;
    Mat right_;
    Mat disparity_;
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

Pipeline::Impl *Pipeline::impl() noexcept
{
    return std::launder(reinterpret_cast<Impl *>(&buffer_));
}

const Pipeline::Impl *Pipeline::impl() const noexcept
{
    return std::launder(reinterpret_cast<const Impl *>(&buffer_));
}

Status Pipeline::Initialize(const std::size_t height, const std::size_t width) noexcept
{
    return this->impl()->Initialize(height, width);
}

Status Pipeline::Run(const void *leftData, const std::size_t leftStride, const void *rightData, const std::size_t rightStride, void *disparityData, const std::size_t disparityStride) const noexcept
{
    return this->impl()->Run(leftData, leftStride, rightData, rightStride, disparityData, disparityStride);
}
} // namespace retinify