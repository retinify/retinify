// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/// @brief
/// Defines a macro for setting API visibility to "default" for the retinify library.
#define RETINIFY_API __attribute__((visibility("default")))

/// @brief
/// Defines a macro to mark functions as deprecated with a custom message.
#define RETINIFY_DEPRECATED(message) __attribute__((deprecated(message)))
