// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace retinify
{
/// @brief
/// Logging verbosity levels for retinify.
enum class LogLevel : std::uint8_t
{
    /// Debug messages.
    DEBUG,
    /// Informational messages.
    INFO,
    /// Warning messages.
    WARN,
    /// Error messages.
    ERROR,
    /// Fatal Error messages.
    FATAL,
    /// Disable all logging.
    OFF,
};

/// @brief
/// Status categories used by retinify.
enum class StatusCategory : std::uint8_t
{
    /// No category.
    NONE,
    /// Retinify-internal category.
    RETINIFY,
    /// System-related category.
    SYSTEM,
    /// CUDA-related category.
    CUDA,
    /// User-originated category.
    USER,
};

/// @brief
/// Status codes returned by retinify operations.
enum class StatusCode : std::uint8_t
{
    /// Operation succeeded.
    OK,
    /// Operation failed.
    FAIL,
    /// Invalid argument provided.
    INVALID_ARGUMENT,
};

/// @brief
/// The mode options for the stereo matching pipeline.
enum class Mode : std::uint8_t
{
    /// Fastest, with lowest accuracy.
    FAST,
    /// Balanced, with moderate accuracy and speed.
    BALANCED,
    /// Most accurate, with slowest performance.
    ACCURATE,
};
} // namespace retinify