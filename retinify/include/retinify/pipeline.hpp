// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/mat.hpp"
#include "retinify/status.hpp"
#include <memory>

namespace retinify
{
class RETINIFY_API Pipeline
{
  public:
    Pipeline() noexcept;
    ~Pipeline() noexcept;
    Pipeline(const Pipeline &) = delete;
    Pipeline &operator=(const Pipeline &) = delete;
    Pipeline(Pipeline &&) noexcept = delete;
    Pipeline &operator=(Pipeline &&) noexcept = delete;
    Status Initialize() const noexcept;
    Status Forward(const Mat &left, const Mat &right, const Mat &disparity) const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace retinify