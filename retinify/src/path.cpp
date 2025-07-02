// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/path.hpp"
#include "retinify/libretinify_onnx.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace retinify
{
constexpr int PATH_ELEMENTS_SIZE = 512;

inline static auto MergePaths(const char *input1, const char *input2) -> const char *
{
    thread_local static std::string buffer;

    if (input1 == nullptr || input2 == nullptr)
    {
        return nullptr;
    }

    std::string_view sv1(input1);
    std::string_view sv2(input2);

    if (sv1.empty() && sv2.empty())
    {
        buffer.clear();
        return buffer.c_str();
    }

    std::filesystem::path path1(sv1);
    std::filesystem::path path2(sv2);
    std::filesystem::path result = path1 / path2;

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

    const bool alreadyExists = std::filesystem::exists(path, error);
    if (error)
    {
        return false;
    }

    if (alreadyExists)
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
            (void)CreateDirectory(path);
            return path;
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
            (void)CreateDirectory(path);
            return path;
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
            (void)CreateDirectory(path);
            return path;
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
            (void)CreateDirectory(path);
            return path;
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

auto TensorRTEngineFilePath() noexcept -> const char *
{
    try
    {
        const char *path = MergePaths(CacheDirectoryPath(), "model.trt");
        if (path != nullptr && std::strlen(path) > 0)
        {
            return path;
        }
    }
    catch (...)
    {
        return nullptr;
    }
    return nullptr;
}
} // namespace retinify