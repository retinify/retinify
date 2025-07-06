// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/path.hpp"
#include "retinify/libretinify_onnx.hpp"
#include "retinify/version.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace retinify
{
inline static auto MergePaths(const char *input1, const char *input2) -> const char *
{
    thread_local static std::string buffer;

    if (input1 == nullptr || input2 == nullptr)
    {
        return nullptr;
    }

    std::string_view stringInput1{input1};
    std::string_view stringInpu2{input2};
    std::string_view stringVersion{Version()};

    if (stringInput1.empty() && stringInpu2.empty())
    {
        buffer.clear();
        return buffer.c_str();
    }

    std::filesystem::path pathInput1{stringInput1};
    std::filesystem::path pathInput2{stringInpu2};
    std::filesystem::path pathVersion{stringVersion};
    std::filesystem::path result = pathInput1 / pathInput2 / pathVersion;
    buffer = result.string();

    return buffer.c_str();
}

inline static auto CreateDirectory(const char *path) -> bool
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
    return nullptr;
}

auto ConfigDirectoryPath() noexcept -> const char *
{
    try
    {
        const char *path = MergePaths(HomeDirectoryPath(), ".config/retinify");
        if (path != nullptr && std::strlen(path) > 0)
        {
            if (CreateDirectory(path))
            {
                return path;
            }
        }
    }
    catch (...)
    {
        return nullptr;
    }
    return nullptr;
}

auto CacheDirectoryPath() noexcept -> const char *
{
    try
    {
        const char *path = MergePaths(HomeDirectoryPath(), ".cache/retinify");
        if (path != nullptr && std::strlen(path) > 0)
        {
            if (CreateDirectory(path))
            {
                return path;
            }
        }
    }
    catch (...)
    {
        return nullptr;
    }
    return nullptr;
}

auto DataDirectoryPath() noexcept -> const char *
{
    try
    {
        const char *path = MergePaths(HomeDirectoryPath(), ".local/share/retinify");
        if (path != nullptr && std::strlen(path) > 0)
        {
            if (CreateDirectory(path))
            {
                return path;
            }
        }
    }
    catch (...)
    {
        return nullptr;
    }
    return nullptr;
}

auto StateDirectoryPath() noexcept -> const char *
{
    try
    {
        const char *path = MergePaths(HomeDirectoryPath(), ".local/state/retinify");
        if (path != nullptr && std::strlen(path) > 0)
        {
            if (CreateDirectory(path))
            {
                return path;
            }
        }
    }
    catch (...)
    {
        return nullptr;
    }
    return nullptr;
}

auto ONNXModelFilePath() noexcept -> const char *
{
    return LIBRETINIFY_ONNX_PATH;
}
} // namespace retinify