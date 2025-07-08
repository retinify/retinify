// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/status.hpp"

#include <cstddef>
#include <type_traits>

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
    Status Initialize(const std::size_t height, const std::size_t width) noexcept;
    Status Run(const void *leftData, const std::size_t leftStride, const void *rightData, const std::size_t rightStride, void *disparityData, const std::size_t disparityStride) const noexcept;

  private:
    class Impl;
    Impl *impl() noexcept;
    const Impl *impl() const noexcept;
    alignas(alignof(std::max_align_t)) unsigned char buffer_[512];
};
} // namespace retinify