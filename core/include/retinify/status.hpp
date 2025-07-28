// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/define.hpp"
#include "retinify/enum.hpp"

namespace retinify
{
/// @brief Status class representing the result of an operation.
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

    /// @brief Check if the status is OK.
    /// @return True if the status is OK, false otherwise.
    [[nodiscard]] auto IsOK() const noexcept -> bool;

    /// @brief Get the status category.
    /// @return The category of the status.
    [[nodiscard]] auto Category() const noexcept -> StatusCategory;

    /// @brief Get the status code.
    /// @return The code of the status.
    [[nodiscard]] auto Code() const noexcept -> StatusCode;

  private:
    StatusCategory category_{StatusCategory::NONE};
    StatusCode code_{StatusCode::OK};
};
} // namespace retinify