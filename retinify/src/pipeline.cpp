// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/pipeline.hpp"
#include "retinify/log.hpp"
#include "retinify/mat.hpp"
#include "retinify/path.hpp"
#include "retinify/status.hpp"
#include "session.hpp"
#include <atomic>
#include <chrono>
#include <iostream>

namespace retinify
{
class Pipeline::Impl
{
  public:
    Impl() = default;
    ~Impl() = default;

    Status Initialize() noexcept
    {
        Status status;

        status = session_.Initialize(ONNXModelFilePath());
        if (!status.IsOK())
        {
            return status;
        }

        initialized_ = true;
        return status;
    }

    Status StereoMatching(const Mat &left, const Mat &right, const Mat &disparity) noexcept
    {
        Status status;

        if (!initialized_)
        {
            status = this->Initialize();
            if (!status.IsOK())
            {
                return status;
            }
        }

        status = left.Wait();
        if (!status.IsOK())
        {
            return status;
        }

        status = right.Wait();
        if (!status.IsOK())
        {
            return status;
        }

        status = disparity.Wait();
        if (!status.IsOK())
        {
            return status;
        }

        status = session_.BindInput("left", left);
        if (!status.IsOK())
        {
            return status;
        }

        status = session_.BindInput("right", right);
        if (!status.IsOK())
        {
            return status;
        }

        status = session_.BindOutput("disparity", disparity);
        if (!status.IsOK())
        {
            return status;
        }

        status = session_.Run();
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

Status Pipeline::Initialize() const noexcept
{
    return this->impl_->Initialize();
}

Status Pipeline::Forward(const Mat &left, const Mat &right, const Mat &disparity) const noexcept
{
    return this->impl_->StereoMatching(left, right, disparity);
}
} // namespace retinify