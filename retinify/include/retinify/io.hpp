// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-EULA

#pragma once

#include "retinify/geometry.hpp"
#include "retinify/status.hpp"

namespace retinify
{
/// @brief
/// Save stereo calibration parameters.
/// @param filename
/// Path to the output file.
/// @param parameters
/// Calibration parameters to save.
/// @return
/// A Status object indicating whether the operation was successful.
RETINIFY_API auto SaveCalibrationParameters(const char *filename, const CalibrationParameters &parameters) noexcept -> Status;

/// @brief
/// Load stereo calibration parameters.
/// @param filename
/// Path to the input file.
/// @param parameters
/// Calibration parameters to load into.
/// @return
/// A Status object indicating whether the operation was successful.
RETINIFY_API auto LoadCalibrationParameters(const char *filename, CalibrationParameters &parameters) noexcept -> Status;
} // namespace retinify
