// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-eula

#include "retinify/paths.hpp"
#include "retinify/logging.hpp"
#include "retinify/retinify_onnx.hpp"
#include "retinify/version.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>

namespace retinify
{
namespace
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

static inline auto ResolveUserDirectory(const char *relativePath, const char *errorMessage) noexcept -> const char *
{
    try
    {
        const char *homeDirectory = HomeDirectoryPath();
        if (homeDirectory == nullptr)
        {
            return nullptr;
        }

        const char *fullPath = JoinPathWithVersion(homeDirectory, relativePath);
        if (fullPath == nullptr || std::strlen(fullPath) == 0)
        {
            return nullptr;
        }

        if (CreateDirectory(fullPath))
        {
            return fullPath;
        }

        LogError(errorMessage);
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
} // namespace

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
    return ResolveUserDirectory(kConfigDirName, "Failed to create or access the configuration directory.");
}

auto CacheDirectoryPath() noexcept -> const char *
{
    return ResolveUserDirectory(kCacheDirName, "Failed to create or access the cache directory.");
}

auto DataDirectoryPath() noexcept -> const char *
{
    return ResolveUserDirectory(kDataDirName, "Failed to create or access the data directory.");
}

auto StateDirectoryPath() noexcept -> const char *
{
    return ResolveUserDirectory(kStateDirName, "Failed to create or access the state directory.");
}

auto ONNXModelFilePath() noexcept -> const char *
{
    return LIBRETINIFY_ONNX_PATH;
}
} // namespace retinify
