// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/path.hpp"
#include "retinify/libretinify_onnx.hpp"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>

#include <sys/stat.h>
#include <unistd.h>

namespace retinify
{
constexpr mode_t DirectoryPermission = 0755;

inline static auto CreateDirectory(const char *path) noexcept -> bool
{
    if (path == nullptr || *path == '\0')
    {
        return false;
    }

    if (::mkdir(path, DirectoryPermission) == 0)
    {
        return true;
    }

    if (errno == EEXIST)
    {
        struct stat status{};
        if (::lstat(path, &status) == 0)
        {
            return S_ISDIR(status.st_mode);
        }
    }

    return false;
}

constexpr int PATH_ELEMENTS_SIZE = 512;

inline static auto MergePaths(const char *path1, const char *path2) noexcept -> const char *
{
    static std::array<char, PATH_ELEMENTS_SIZE> buffer{};
    static const char *separator{"/"};

    if (path1 == nullptr || path2 == nullptr)
    {
        return nullptr;
    }

    const std::size_t lenPath1 = std::strlen(path1);
    const std::size_t lenPath2 = std::strlen(path2);

    bool needsSeparator = (lenPath1 > 0 && path1[lenPath1 - 1] != *separator) && (lenPath2 > 0 && path2[0] != *separator);

    std::size_t total_len = lenPath1 + lenPath2 + (needsSeparator ? 1 : 0);
    if (total_len + 1 >= buffer.size())
    {
        return nullptr;
    }

    std::strcpy(buffer.data(), path1);
    if (needsSeparator)
    {
        std::strcat(buffer.data(), separator);
    }
    std::strcat(buffer.data(), path2);

    return buffer.data();
}

auto HomeDirectoryPath() noexcept -> const char *
{
    const char *home_path = std::getenv("HOME");
    if (home_path != nullptr && std::strlen(home_path) > 0)
    {
        return home_path;
    }

    return nullptr;
}

auto ConfigDirectoryPath() noexcept -> const char *
{
    const char *config_path = MergePaths(HomeDirectoryPath(), ".config/retinify");
    if (config_path == nullptr || std::strlen(config_path) == 0)
    {
        return nullptr;
    }

    (void)CreateDirectory(config_path);

    return config_path;
}

auto CacheDirectoryPath() noexcept -> const char *
{
    const char *cache_path = MergePaths(HomeDirectoryPath(), ".cache/retinify");
    if (cache_path == nullptr || std::strlen(cache_path) == 0)
    {
        return nullptr;
    }

    (void)CreateDirectory(cache_path);

    return cache_path;
}

auto DataDirectoryPath() noexcept -> const char *
{
    const char *data_path = MergePaths(HomeDirectoryPath(), ".local/share/retinify");
    if (data_path == nullptr || std::strlen(data_path) == 0)
    {
        return nullptr;
    }

    (void)CreateDirectory(data_path);

    return data_path;
}

auto StateDirectoryPath() noexcept -> const char *
{
    const char *state_path = MergePaths(HomeDirectoryPath(), ".local/state/retinify");
    if (state_path == nullptr || std::strlen(state_path) == 0)
    {
        return nullptr;
    }

    (void)CreateDirectory(state_path);

    return state_path;
}

RETINIFY_API auto ONNXModelFilePath() noexcept -> const char *
{
    return LIBRETINIFY_ONNX_PATH;
}

RETINIFY_API auto TensorRTEngineFilePath() noexcept -> const char *
{
    const char *trt_engine_path = MergePaths(CacheDirectoryPath(), "model.trt");
    if (trt_engine_path == nullptr || std::strlen(trt_engine_path) == 0)
    {
        return nullptr;
    }

    return trt_engine_path;
}
} // namespace retinify