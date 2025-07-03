// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/status.hpp"

#include <sstream>

namespace retinify
{
Status::Status(const StatusCategory category, const StatusCode code) noexcept : category_(category), code_(code)
{
}

auto Status::IsOK() const noexcept -> bool
{
    return code_ == StatusCode::OK;
}

auto Status::Category() const noexcept -> StatusCategory
{
    return category_;
}

auto Status::Code() const noexcept -> StatusCode
{
    return code_;
}
} // namespace retinify