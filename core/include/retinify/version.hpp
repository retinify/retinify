// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "retinify/define.hpp"

namespace retinify
{
/// @brief
/// Returns the Retinify library version in semantic versioning format.
/// @return
/// A string representing the Retinify library version.
RETINIFY_API auto Version() noexcept -> const char *;
} // namespace retinify