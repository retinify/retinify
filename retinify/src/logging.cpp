// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-EULA

#include "retinify/logging.hpp"

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace retinify
{
static inline auto GetLogLevelRef() noexcept -> LogLevel &
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

namespace
{
constexpr const char *kDefaultLabel = "NONE ";
constexpr const char *kDefaultMessage = "No message provided.";

struct LogMetadata
{
    const char *label;
    const char *colorCode;
    std::ostream *destination;
};

[[nodiscard]] auto GetMetadataForLevel(LogLevel level) noexcept -> LogMetadata
{
    switch (level)
    {
    case LogLevel::DEBUG:
        return {"DEBUG", "\033[35m", &std::cout};
    case LogLevel::INFO:
        return {"INFO ", "\033[32m", &std::cout};
    case LogLevel::WARN:
        return {"WARN ", "\033[33m", &std::cerr};
    case LogLevel::ERROR:
        return {"ERROR", "\033[31m", &std::cerr};
    case LogLevel::FATAL:
        return {"FATAL", "\033[31;1m", &std::cerr};
    case LogLevel::OFF:
    default:
        return {kDefaultLabel, "\033[0m", &std::cerr};
    }
}

[[nodiscard]] auto ShouldLog(LogLevel level) noexcept -> bool
{
    return static_cast<int>(level) >= static_cast<int>(GetLogLevel());
}

[[nodiscard]] auto SanitizeMessage(const char *message) noexcept -> const char *
{
    if (message == nullptr || std::strlen(message) == 0)
    {
        return kDefaultMessage;
    }

    return message;
}

static inline auto GetCurrentTime() -> std::string
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
} // namespace

static inline void Log(LogLevel level, const char *message, std::source_location location) noexcept
{
    if (!ShouldLog(level))
    {
        return;
    }

    try
    {
        const LogMetadata metadata = GetMetadataForLevel(level);
        if (metadata.destination == nullptr)
        {
            return;
        }

        std::ostream &out = *metadata.destination;
        if (!out.good())
        {
            return;
        }

        const char *functionName = location.function_name();
        if (functionName == nullptr)
        {
            return;
        }

        out << "[" << GetCurrentTime() << "]"                                  //
            << "[" << metadata.colorCode << metadata.label << "\033[0m" << "]" //
            << "[" << functionName << "]"                                      //
            << SanitizeMessage(message) << '\n';                               //
    }
    catch (...) // NOLINT(bugprone-empty-catch)
    {
        // do nothing
    }
}

void LogDebug(const char *message, const std::source_location location) noexcept
{
    Log(LogLevel::DEBUG, message, location);
}

void LogInfo(const char *message, const std::source_location location) noexcept
{
    Log(LogLevel::INFO, message, location);
}

void LogWarn(const char *message, const std::source_location location) noexcept
{
    Log(LogLevel::WARN, message, location);
}

void LogError(const char *message, const std::source_location location) noexcept
{
    Log(LogLevel::ERROR, message, location);
}

void LogFatal(const char *message, const std::source_location location) noexcept
{
    Log(LogLevel::FATAL, message, location);
}
} // namespace retinify
