// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/path.hpp"
#include "retinify/log.hpp"
#include "retinify/retinify_onnx.hpp"
#include "retinify/version.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>

namespace retinify
{
constexpr auto kConfigDirName = ".config/retinify";
constexpr auto kCacheDirName = ".cache/retinify";
constexpr auto kDataDirName = ".local/share/retinify";
constexpr auto kStateDirName = ".local/state/retinify";

static inline auto JoinPathWithVersion(const char *input1, const char *input2) -> const char *
{
    if (input1 == nullptr || input2 == nullptr)
    {
        return nullptr;
    }

    thread_local static std::string buffer;

    std::filesystem::path result = std::filesystem::path(input1) / input2 / Version();
    buffer = result.string();

    return buffer.c_str();
}

static inline auto CreateDirectory(const char *path) -> bool
{
    if (path == nullptr || std::strlen(path) == 0)
    {
        return false;
    }

    std::error_code error;

    const bool exists = std::filesystem::exists(path, error);
    if (error)
    {
        return false;
    }

    if (exists)
    {
        const bool isDir = std::filesystem::is_directory(path, error);
        return !error && isDir;
    }

    std::filesystem::create_directories(path, error);
    return !error;
}

auto HomeDirectoryPath() noexcept -> const char *
{
    const char *path = std::getenv("HOME");
    if (path != nullptr && std::strlen(path) > 0)
    {
        return path;
    }

    LogError("Environment variable 'HOME' is not set or empty.");
    return nullptr;
}

auto ConfigDirectoryPath() noexcept -> const char *
{
    try
    {
        const char *path = JoinPathWithVersion(HomeDirectoryPath(), kConfigDirName);
        if (path != nullptr && std::strlen(path) > 0)
        {
            if (CreateDirectory(path))
            {
                return path;
            }

            LogError("Failed to create or access the configuration directory.");
        }
    }
    catch (std::exception &e)
    {
        LogError(e.what());
    }
    catch (...)
    {
        LogFatal("An unknown error occurred.");
    }

    return nullptr;
}

auto CacheDirectoryPath() noexcept -> const char *
{
    try
    {
        const char *path = JoinPathWithVersion(HomeDirectoryPath(), kCacheDirName);
        if (path != nullptr && std::strlen(path) > 0)
        {
            if (CreateDirectory(path))
            {
                return path;
            }

            LogError("Failed to create or access the cache directory.");
        }
    }
    catch (std::exception &e)
    {
        LogError(e.what());
    }
    catch (...)
    {
        LogFatal("An unknown error occurred.");
    }

    return nullptr;
}

auto DataDirectoryPath() noexcept -> const char *
{
    try
    {
        const char *path = JoinPathWithVersion(HomeDirectoryPath(), kDataDirName);
        if (path != nullptr && std::strlen(path) > 0)
        {
            if (CreateDirectory(path))
            {
                return path;
            }

            LogError("Failed to create or access the data directory.");
        }
    }
    catch (std::exception &e)
    {
        LogError(e.what());
    }
    catch (...)
    {
        LogFatal("An unknown error occurred.");
    }

    return nullptr;
}

auto StateDirectoryPath() noexcept -> const char *
{
    try
    {
        const char *path = JoinPathWithVersion(HomeDirectoryPath(), kStateDirName);
        if (path != nullptr && std::strlen(path) > 0)
        {
            if (CreateDirectory(path))
            {
                return path;
            }

            LogError("Failed to create or access the state directory.");
        }
    }
    catch (std::exception &e)
    {
        LogError(e.what());
    }
    catch (...)
    {
        LogFatal("An unknown error occurred.");
    }

    return nullptr;
}

auto ONNXModelFilePath() noexcept -> const char *
{
    return LIBRETINIFY_ONNX_PATH;
}
} // namespace retinify