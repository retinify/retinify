// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/define.hpp"
#include "retinify/enum.hpp"

namespace retinify
{
class RETINIFY_API Status
{
  public:
    Status() noexcept;
    explicit Status(StatusCategory category, StatusCode code) noexcept;
    ~Status() = default;
    Status(const Status &) noexcept = default;
    auto operator=(const Status &) noexcept -> Status & = default;
    Status(Status &&) noexcept = default;
    auto operator=(Status &&) noexcept -> Status & = default;
    [[nodiscard]] auto IsOK() const noexcept -> bool;
    [[nodiscard]] auto Category() const noexcept -> StatusCategory;
    [[nodiscard]] auto Code() const noexcept -> StatusCode;

  private:
    StatusCategory category_;
    StatusCode code_;
};
} // namespace retinify