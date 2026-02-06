// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/attributes.hpp"
#include "retinify/status.hpp"

#include <cstdint>
#include <source_location>

namespace retinify
{
/// @brief
/// Logging verbosity levels
enum class LogLevel : std::uint8_t
{
    /// @brief
    /// Debug messages
    DEBUG,
    /// @brief
    /// Informational messages
    INFO,
    /// @brief
    /// Warning messages
    WARN,
    /// @brief
    /// Error messages
    ERROR,
    /// @brief
    /// Fatal Error messages
    FATAL,
    /// @brief
    /// Disable all logging
    OFF,
};

/// @brief
/// Returns the current log level
/// @return
/// The current log level
RETINIFY_API auto GetLogLevel() noexcept -> LogLevel;

/// @brief
/// Sets the log level
/// @param level
/// The new log level to apply
RETINIFY_API auto SetLogLevel(LogLevel level) noexcept -> void;

/// @brief
/// Logging source location options
enum class LogLocation : std::uint8_t
{
    /// @brief
    /// No source location
    NONE,
    /// @brief
    /// Function name
    FUNCTION,
};

/// @brief
/// Returns the current log location setting
/// @return
/// The current log location setting
RETINIFY_API auto GetLogLocation() noexcept -> LogLocation;

/// @brief
/// Sets the log location setting
/// @param location
/// The new log location setting to apply
RETINIFY_API auto SetLogLocation(LogLocation location) noexcept -> void;

/// @brief
/// Logs a debug message
/// @param message
/// The message to log
/// @param location
/// The source location of the log call (defaults to the call site)
RETINIFY_API auto LogDebug(const char *message, std::source_location location = std::source_location::current()) noexcept -> void;

/// @brief
/// Logs an informational message
/// @param message
/// The message to log
/// @param location
/// The source location of the log call (defaults to the call site)
RETINIFY_API auto LogInfo(const char *message, std::source_location location = std::source_location::current()) noexcept -> void;

/// @brief
/// Logs a warning message
/// @param message
/// The message to log
/// @param location
/// The source location of the log call (defaults to the call site)
RETINIFY_API auto LogWarn(const char *message, std::source_location location = std::source_location::current()) noexcept -> void;

/// @brief
/// Logs an error message
/// @param message
/// The message to log
/// @param location
/// The source location of the log call (defaults to the call site)
RETINIFY_API auto LogError(const char *message, std::source_location location = std::source_location::current()) noexcept -> void;

/// @brief
/// Logs a fatal error message
/// @param message
/// The message to log
/// @param location
/// The source location of the log call (defaults to the call site)
RETINIFY_API auto LogFatal(const char *message, std::source_location location = std::source_location::current()) noexcept -> void;

/// @brief
/// Logs basic information about the retinify library
/// @param location
/// The source location of the log call (defaults to the call site)
RETINIFY_API auto LogSoftwareInfo(std::source_location location = std::source_location::current()) noexcept -> void;

/// @brief
/// Logs an error indicating that the provided stride is smaller than the required stride
/// @param providedStride
/// The provided stride value (in bytes)
/// @param requiredStride
/// The required stride value (in bytes)
/// @param location
/// The source location of the log call (defaults to the call site)
RETINIFY_API auto LogStrideError(std::size_t providedStride, std::size_t requiredStride, std::source_location location = std::source_location::current()) noexcept -> void;
} // namespace retinify
