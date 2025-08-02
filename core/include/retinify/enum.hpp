// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace retinify
{
/// @brief Logging verbosity levels for retinify.
enum class LogLevel : std::uint8_t
{
    /// Debug-level messages.
    DEBUG,
    /// Informational messages.
    INFO,
    /// Warning-level messages.
    WARN,
    /// Error-level messages.
    ERROR,
    /// Critical-level messages.
    FATAL,
    /// Disable all logging.
    OFF,
};

/// @brief Status categories used by retinify.
enum class StatusCategory : std::uint8_t
{
    /// No status category.
    NONE,
    /// retinify-specific status codes.
    RETINIFY,
    /// System-related status codes.
    SYSTEM,
    /// CUDA-related status codes.
    CUDA,
    /// User-originated status codes.
    USER,
};

/// @brief Status codes returned by retinify operations.
enum class StatusCode : std::uint8_t
{
    /// Operation succeeded.
    OK,
    /// Operation failed.
    FAIL,
    /// Invalid argument provided.
    INVALID_ARGUMENT,
};
} // namespace retinify