// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/define.hpp"
#include "retinify/enum.hpp"
#include "retinify/status.hpp"

#include <source_location>

namespace retinify
{
RETINIFY_API auto GetLogLevel() noexcept -> LogLevel;
RETINIFY_API void SetLogLevel(LogLevel level) noexcept;
RETINIFY_API void LogDebug(const char *message, std::source_location location = std::source_location::current()) noexcept;
RETINIFY_API void LogInfo(const char *message, std::source_location location = std::source_location::current()) noexcept;
RETINIFY_API void LogWarn(const char *message, std::source_location location = std::source_location::current()) noexcept;
RETINIFY_API void LogError(const char *message, std::source_location location = std::source_location::current()) noexcept;
RETINIFY_API void LogFatal(const char *message, std::source_location location = std::source_location::current()) noexcept;
} // namespace retinify
