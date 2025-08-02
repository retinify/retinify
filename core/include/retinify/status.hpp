// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/define.hpp"
#include "retinify/enum.hpp"

namespace retinify
{
/// @brief
/// This class represents the status of an operation in the retinify library.
class RETINIFY_API Status
{
  public:
    Status() noexcept = default;
    explicit Status(StatusCategory category, StatusCode code) noexcept;
    ~Status() noexcept = default;
    Status(const Status &) noexcept = default;
    auto operator=(const Status &) noexcept -> Status & = default;
    Status(Status &&) noexcept = default;
    auto operator=(Status &&) noexcept -> Status & = default;

    /// @brief
    /// Returns whether the status is OK.
    /// @return
    /// True if the status is OK; false otherwise.
    [[nodiscard]] auto IsOK() const noexcept -> bool;

    /// @brief
    /// Returns the status category.
    /// @return
    /// The status category.
    [[nodiscard]] auto Category() const noexcept -> StatusCategory;

    /// @brief
    /// Returns the status code.
    /// @return
    /// The status code.
    [[nodiscard]] auto Code() const noexcept -> StatusCode;

  private:
    StatusCategory category_{StatusCategory::NONE};
    StatusCode code_{StatusCode::OK};
};
} // namespace retinify