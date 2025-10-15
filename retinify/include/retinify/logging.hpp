// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-eula

#pragma once

#include "retinify/attributes.hpp"
#include "retinify/status.hpp"

#include <cstdint>
#include <source_location>

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
/// Returns the current log level.
/// @return
/// The current log level.
RETINIFY_API auto GetLogLevel() noexcept -> LogLevel;

/// @brief
/// Sets the log level.
/// @param level
/// The new log level to apply.
RETINIFY_API void SetLogLevel(LogLevel level) noexcept;

/// @brief
/// Logs a debug message.
/// @param message
/// The message to log.
/// @param location
/// The source location of the log call (defaults to the call site).
RETINIFY_API void LogDebug(const char *message, std::source_location location = std::source_location::current()) noexcept;

/// @brief
/// Logs an informational message.
/// @param message
/// The message to log.
/// @param location
/// The source location of the log call (defaults to the call site).
RETINIFY_API void LogInfo(const char *message, std::source_location location = std::source_location::current()) noexcept;

/// @brief
/// Logs a warning message.
/// @param message
/// The message to log.
/// @param location
/// The source location of the log call (defaults to the call site).
RETINIFY_API void LogWarn(const char *message, std::source_location location = std::source_location::current()) noexcept;

/// @brief
/// Logs an error message.
/// @param message
/// The message to log.
/// @param location
/// The source location of the log call (defaults to the call site).
RETINIFY_API void LogError(const char *message, std::source_location location = std::source_location::current()) noexcept;

/// @brief
/// Logs a fatal error message.
/// @param message
/// The message to log.
/// @param location
/// The source location of the log call (defaults to the call site).
RETINIFY_API void LogFatal(const char *message, std::source_location location = std::source_location::current()) noexcept;
} // namespace retinify
