// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace retinify
{
enum class LogLevel : std::uint8_t
{
    DEBUG,
    INFO,
    WARN,
    ERROR,
    FATAL,
    OFF,
};

enum class StatusCategory : std::uint8_t
{
    NONE,
    DEVICE,
    SYSTEM,
    RETINIFY,
    CUDA,
    USER,
};

enum class StatusCode : std::uint8_t
{
    OK,
    FAIL,
    INVALID_ARGUMENT,
    NOT_ALLOCATED,
    NULL_POINTER,
};
} // namespace retinify