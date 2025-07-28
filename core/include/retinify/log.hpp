// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/define.hpp"
#include "retinify/enum.hpp"
#include "retinify/status.hpp"

#include <source_location>

namespace retinify
{
/// @brief Get the current log level.
/// @return The current log level.
RETINIFY_API auto GetLogLevel() noexcept -> LogLevel;

/// @brief Set the log level.
/// @param level The log level to set.
RETINIFY_API void SetLogLevel(LogLevel level) noexcept;

/// @brief Log a debug message.
/// @param message The message to log.
/// @param location The source location of the log call, defaults to the current location.
RETINIFY_API void LogDebug(const char *message, std::source_location location = std::source_location::current()) noexcept;

/// @brief Log an informational message.
/// @param message The message to log.
/// @param location The source location of the log call, defaults to the current location.
RETINIFY_API void LogInfo(const char *message, std::source_location location = std::source_location::current()) noexcept;

/// @brief Log a warning message.
/// @param message The message to log.
/// @param location The source location of the log call, defaults to the current location.
RETINIFY_API void LogWarn(const char *message, std::source_location location = std::source_location::current()) noexcept;

/// @brief Log an error message.
/// @param message The message to log.
/// @param location The source location of the log call, defaults to the current location.
RETINIFY_API void LogError(const char *message, std::source_location location = std::source_location::current()) noexcept;

/// @brief Log a fatal error message.
/// @param message The message to log.
/// @param location The source location of the log call, defaults to the current location.
RETINIFY_API void LogFatal(const char *message, std::source_location location = std::source_location::current()) noexcept;
} // namespace retinify
