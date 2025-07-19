// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/define.hpp"

namespace retinify
{
RETINIFY_API auto HomeDirectoryPath() noexcept -> const char *;
RETINIFY_API auto ConfigDirectoryPath() noexcept -> const char *;
RETINIFY_API auto CacheDirectoryPath() noexcept -> const char *;
RETINIFY_API auto DataDirectoryPath() noexcept -> const char *;
RETINIFY_API auto StateDirectoryPath() noexcept -> const char *;
RETINIFY_API auto ONNXModelFilePath() noexcept -> const char *;
} // namespace retinify