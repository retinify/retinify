// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/log.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

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

inline static auto GetCurrentTime() -> std::string
{
    const std::chrono::system_clock::time_point currentTimePoint = std::chrono::system_clock::now();
    const std::time_t currentTimeT = std::chrono::system_clock::to_time_t(currentTimePoint);
    std::tm localTimeStruct{};
    if (localtime_r(&currentTimeT, &localTimeStruct) == nullptr)
    {
        return std::string{"Invalid time"};
    }
    std::ostringstream timeStream;
    timeStream << std::put_time(&localTimeStruct, "%F %T");
    return timeStream.str();
}

inline static auto GetColorCode(LogLevel level) noexcept -> const char *
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
        return "\033[0m";
    }
}

inline static void Log(LogLevel level, const char *label, const char *message, std::ostream &out, std::source_location location) noexcept
{
    if (static_cast<int>(level) < static_cast<int>(GetLogLevel()))
    {
        return;
    }

    try
    {
        if (label == nullptr || std::strlen(label) == 0)
        {
            label = "NONE ";
        }

        if (message == nullptr || std::strlen(message) == 0)
        {
            message = "No message provided.";
        }

        if (location.function_name() == nullptr)
        {
            return;
        }

        if (!out.good())
        {
            return;
        }

        out << "[" << GetCurrentTime() << "]"                          //
            << "[" << GetColorCode(level) << label << "\033[0m" << "]" //
            << "[" << location.function_name() << "]"                  //
            << message << '\n';                                        //
    }
    catch (...) // NOLINT(bugprone-empty-catch)
    {
        // do nothing
    }
}

void LogDebug(const char *message, const std::source_location location) noexcept
{
    Log(LogLevel::DEBUG, "DEBUG", message, std::cout, location);
}

void LogInfo(const char *message, const std::source_location location) noexcept
{
    Log(LogLevel::INFO, "INFO ", message, std::cout, location);
}

void LogWarn(const char *message, const std::source_location location) noexcept
{
    Log(LogLevel::WARN, "WARN ", message, std::cerr, location);
}

void LogError(const char *message, const std::source_location location) noexcept
{
    Log(LogLevel::ERROR, "ERROR", message, std::cerr, location);
}

void LogFatal(const char *message, const std::source_location location) noexcept
{
    Log(LogLevel::FATAL, "FATAL", message, std::cerr, location);
}
} // namespace retinify