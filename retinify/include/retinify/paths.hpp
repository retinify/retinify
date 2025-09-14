// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/attributes.hpp"

namespace retinify
{
/// @brief Returns the current user’s home directory path.
/// @return A null-terminated string with the home directory path, or nullptr on failure.
RETINIFY_API auto HomeDirectoryPath() noexcept -> const char *;

/// @brief Returns the configuration directory path for retinify.
/// @return A null-terminated string with the config directory path, or nullptr on failure.
RETINIFY_API auto ConfigDirectoryPath() noexcept -> const char *;

/// @brief Returns the cache directory path for retinify.
/// @return A null-terminated string with the cache directory path, or nullptr on failure.
RETINIFY_API auto CacheDirectoryPath() noexcept -> const char *;

/// @brief Returns the data directory path for retinify.
/// @return A null-terminated string with the data directory path, or nullptr on failure.
RETINIFY_API auto DataDirectoryPath() noexcept -> const char *;

/// @brief Returns the state directory path for retinify.
/// @return A null-terminated string with the state directory path, or nullptr on failure.
RETINIFY_API auto StateDirectoryPath() noexcept -> const char *;

/// @brief Returns the path to the ONNX model file used by retinify.
/// @return A null-terminated string with the ONNX model file path.
RETINIFY_API auto ONNXModelFilePath() noexcept -> const char *;
} // namespace retinify