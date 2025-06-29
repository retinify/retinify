// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/version.hpp"
#include "retinify/libretinify_version.hpp"

namespace retinify
{
auto Version() noexcept -> const char *
{
    return LIBRETINIFY_VERSION;
}
} // namespace retinify