// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mat.hpp"
#include "session.hpp"

#include "retinify/log.hpp"
#include "retinify/path.hpp"
#include "retinify/pipeline.hpp"
#include "retinify/status.hpp"

#include <atomic>
#include <chrono>
#include <iostream>

namespace retinify
{
class Pipeline::Impl
{
  public:
    Impl() = default;

    ~Impl()
    {
        initialized_ = false;
        (void)left_.Free();
        (void)right_.Free();
        (void)disparity_.Free();
    }

    Status Initialize(const std::size_t height, const std::size_t weidth) noexcept
    {
        Status status;

        status = left_.Allocate(height, weidth, 3, sizeof(float));
        if (!status.IsOK())
        {
            return status;
        }

        status = right_.Allocate(height, weidth, 3, sizeof(float));
        if (!status.IsOK())
        {
            return status;
        }

        status = disparity_.Allocate(height, weidth, 1, sizeof(float));
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

    Status StereoMatching(const void *leftData, const std::size_t leftStride, const void *rightData, const std::size_t rightStride, void *disparityData, const std::size_t disparityStride) const noexcept
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

Pipeline::Pipeline() noexcept : impl_(std::make_unique<Impl>())
{
}

Pipeline::~Pipeline() noexcept = default;

Status Pipeline::Initialize(const std::size_t height, const std::size_t weidth) noexcept
{
    return this->impl_->Initialize(height, weidth);
}

Status Pipeline::Forward(const void *leftData, const std::size_t leftStride, const void *rightData, const std::size_t rightStride, void *disparityData, const std::size_t disparityStride) const noexcept
{
    return this->impl_->StereoMatching(leftData, leftStride, rightData, rightStride, disparityData, disparityStride);
}
} // namespace retinify