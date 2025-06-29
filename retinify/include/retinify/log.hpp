// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/define.hpp"
#include "retinify/enum.hpp"
#include <source_location>
#include <string_view>

namespace retinify
{
RETINIFY_API auto GetLogLevel() noexcept -> LogLevel;
RETINIFY_API void SetLogLevel(LogLevel level) noexcept;
RETINIFY_API void LogDebug(std::string_view msg, std::source_location location = std::source_location::current()) noexcept;
RETINIFY_API void LogInfo(std::string_view msg, std::source_location location = std::source_location::current()) noexcept;
RETINIFY_API void LogWarn(std::string_view msg, std::source_location location = std::source_location::current()) noexcept;
RETINIFY_API void LogError(std::string_view msg, std::source_location location = std::source_location::current()) noexcept;
RETINIFY_API void LogFatal(std::string_view msg, std::source_location location = std::source_location::current()) noexcept;
} // namespace retinify
