// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/log.hpp"
#include <atomic>
#include <chrono>
#include <iostream>

namespace retinify
{
inline static auto GetLogLevelRef() noexcept -> LogLevel &
{
    static LogLevel level = LogLevel::INFO;
    return level;
}

auto GetLogLevel() noexcept -> LogLevel
{
    return GetLogLevelRef();
}

void SetLogLevel(LogLevel level) noexcept
{
    GetLogLevelRef() = level;
}

constexpr size_t timeBufferSize = 20;

inline static auto GetCurrentTime() noexcept -> std::array<char, timeBufferSize>
{
    std::array<char, timeBufferSize> timeBuffer{};

    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);

    std::tm result{};
    if (localtime_r(&time, &result) != nullptr)
    {
        const size_t size = std::strftime(timeBuffer.data(), timeBuffer.size(), "%F %T", &result);
        if (size == 0)
        {
            timeBuffer[0] = '\0';
        }
    }

    return timeBuffer;
}

inline static auto GetColorCode(LogLevel level) noexcept -> std::string_view
{
    switch (level)
    {
    case LogLevel::DEBUG:
        return "\033[35m";
    case LogLevel::INFO:
        return "\033[32m";
    case LogLevel::WARN:
        return "\033[33m";
    case LogLevel::ERROR:
        return "\033[31m";
    case LogLevel::FATAL:
        return "\033[31;1m";
    default:
        return "";
    }
}

inline static void Log(LogLevel level, std::string_view label, std::string_view msg, std::ostream &out, std::source_location location) noexcept
{
    if (static_cast<int>(level) < static_cast<int>(GetLogLevel()))
    {
        return;
    }
    auto time = GetCurrentTime();
    out << "[" << time.data() << "]"
        << "[" << GetColorCode(level) << label << "\033[0m" << "]"
        << "[" << location.function_name() << "]" << msg << '\n';
}

void LogDebug(const std::string_view msg, const std::source_location location) noexcept
{
    Log(LogLevel::DEBUG, "DEBUG", msg, std::cout, location);
}

void LogInfo(const std::string_view msg, const std::source_location location) noexcept
{
    Log(LogLevel::INFO, "INFO ", msg, std::cout, location);
}

void LogWarn(const std::string_view msg, const std::source_location location) noexcept
{
    Log(LogLevel::WARN, "WARN ", msg, std::cout, location);
}

void LogError(const std::string_view msg, const std::source_location location) noexcept
{
    Log(LogLevel::ERROR, "ERROR", msg, std::cerr, location);
}

void LogFatal(const std::string_view msg, const std::source_location location) noexcept
{
    Log(LogLevel::FATAL, "FATAL", msg, std::cerr, location);
}
} // namespace retinify