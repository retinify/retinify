// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/logging.hpp"
#include "retinify/version.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <format>
#include <iomanip>
#include <iostream>
#include <source_location>
#include <sstream>

namespace retinify
{
static auto GetLogLevelStorage() noexcept -> std::atomic<LogLevel> &
{
    static std::atomic<LogLevel> storage{LogLevel::INFO};
    return storage;
}

auto GetLogLevel() noexcept -> LogLevel
{
    return GetLogLevelStorage().load(std::memory_order_relaxed);
}

void SetLogLevel(LogLevel level) noexcept
{
    GetLogLevelStorage().store(level, std::memory_order_relaxed);
}

static auto GetLogLocationStorage() noexcept -> std::atomic<LogLocation> &
{
    static std::atomic<LogLocation> storage{LogLocation::NONE};
    return storage;
}

auto GetLogLocation() noexcept -> LogLocation
{
    return GetLogLocationStorage().load(std::memory_order_relaxed);
}

void SetLogLocation(LogLocation location) noexcept
{
    GetLogLocationStorage().store(location, std::memory_order_relaxed);
}

namespace
{
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
        return {"NONE ", "\033[0m", &std::cerr};
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
        return " ";
    }

    return message;
}

[[nodiscard]] auto GetCurrentTime() -> std::string
{
    const std::time_t now = std::time(nullptr);

    std::tm utc{};
#if defined(_WIN32)
    if (gmtime_s(&utc, &now) != 0)
    {
        return {};
    }
#else
    if (gmtime_r(&now, &utc) == nullptr)
    {
        return {};
    }
#endif

    char buffer[32];
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc) == 0)
    {
        return {};
    }

    return buffer;
}

inline auto Log(LogLevel level, const char *message, std::source_location location) noexcept -> void
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

        out << "[" << GetCurrentTime() << "]"

            << "[" << metadata.colorCode << metadata.label << "\033[0m" << "]";

        switch (GetLogLocation())
        {
        case LogLocation::NONE:
            break;
        case LogLocation::FUNCTION:
            out << "[" << location.function_name() << "]";
            break;
        default:
            break;
        }

        out << SanitizeMessage(message) << '\n';
    }
    catch (...) // NOLINT(bugprone-empty-catch)
    {
        // do nothing
    }
}
} // namespace

auto LogDebug(const char *message, const std::source_location location) noexcept -> void
{
    Log(LogLevel::DEBUG, message, location);
}

auto LogInfo(const char *message, const std::source_location location) noexcept -> void
{
    Log(LogLevel::INFO, message, location);
}

auto LogWarn(const char *message, const std::source_location location) noexcept -> void
{
    Log(LogLevel::WARN, message, location);
}

auto LogError(const char *message, const std::source_location location) noexcept -> void
{
    Log(LogLevel::ERROR, message, location);
}

auto LogFatal(const char *message, const std::source_location location) noexcept -> void
{
    Log(LogLevel::FATAL, message, location);
}

auto LogSoftwareInfo(const std::source_location location) noexcept -> void
{
    static std::atomic_flag printed = ATOMIC_FLAG_INIT;
    if (printed.test_and_set())
    {
        return;
    }

    char summaryBuffer[128];
    std::snprintf(summaryBuffer, sizeof(summaryBuffer), "retinify v%s | Real-Time AI Stereo Vision Library | Copyright (c) 2025 Sensui Yagi", Version());
    LogInfo(summaryBuffer, location);
}

auto LogStrideError(std::size_t providedStride, std::size_t requiredStride, const std::source_location location) noexcept -> void
{
    char messageBuffer[256];
    std::snprintf(messageBuffer, sizeof(messageBuffer), "Provided stride (%zu bytes) is smaller than the required stride (%zu bytes).", providedStride, requiredStride);
    LogError(messageBuffer, location);
}
} // namespace retinify
