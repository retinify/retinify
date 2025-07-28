// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/define.hpp"

namespace retinify
{
/// @brief Get the version of the Retinify library in semantic versioning format.
/// @return The version string of the Retinify library.
RETINIFY_API auto Version() noexcept -> const char *;
} // namespace retinify