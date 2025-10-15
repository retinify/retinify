// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-eula

#include "retinify/version.hpp"
#include "retinify/retinify_version.hpp"

namespace retinify
{
auto Version() noexcept -> const char *
{
    return LIBRETINIFY_VERSION;
}
} // namespace retinify