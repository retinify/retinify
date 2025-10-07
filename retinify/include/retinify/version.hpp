// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-EULA

#pragma once

#include "retinify/attributes.hpp"

namespace retinify
{
/// @brief
/// Returns the retinify library version in semantic versioning format.
/// @return
/// A string representing the retinify library version.
RETINIFY_API auto Version() noexcept -> const char *;
} // namespace retinify