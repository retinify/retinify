// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

namespace retinify
{
/// @brief 2D vector (double).
using Vec2d = std::array<double, 2>;

/// @brief 3D vector (double).
using Vec3d = std::array<double, 3>;

/// @brief 2D point (double).
using Point2d = std::array<double, 2>;

/// @brief 3D point (double).
using Point3d = std::array<double, 3>;

/// @brief 3x3 matrix (double, row-major).
using Mat3x3d = std::array<std::array<double, 3>, 3>;

/// @brief 3x4 matrix (double, row-major).
using Mat3x4d = std::array<std::array<double, 4>, 3>;

/// @brief 4x4 matrix (double, row-major).
using Mat4x4d = std::array<std::array<double, 4>, 4>;

/// @brief 2D vector (float).
using Vec2f = std::array<float, 2>;

/// @brief 3D vector (float).
using Vec3f = std::array<float, 3>;

/// @brief 2D point (float).
using Point2f = std::array<float, 2>;

/// @brief 3D point (float).
using Point3f = std::array<float, 3>;

/// @brief 3x3 matrix (float, row-major).
using Mat3x3f = std::array<std::array<float, 3>, 3>;

/// @brief 3x4 matrix (float, row-major).
using Mat3x4f = std::array<std::array<float, 4>, 3>;

/// @brief 4x4 matrix (float, row-major).
using Mat4x4f = std::array<std::array<float, 4>, 4>;

/// @brief
/// Camera intrinsic parameters with focal lengths, principal point, and skew.
struct Intrinsics
{
    double fx{0};   // Focal length in x [pixels]
    double fy{0};   // Focal length in y [pixels]
    double cx{0};   // Principal point x-coordinate [pixels]
    double cy{0};   // Principal point y-coordinate [pixels]
    double skew{0}; // Skew coefficient
};

/// @brief
/// Brown–Conrady distortion model with 5 coefficients (k1, k2, p1, p2, k3).
struct Distortion
{
    double k1{0};
    double k2{0};
    double p1{0};
    double p2{0};
    double k3{0};
};

/// @brief
/// Rational distortion model with 8 coefficients (k1, k2, p1, p2, k3, k4, k5, k6).
struct DistortionRational
{
    double k1{0};
    double k2{0};
    double p1{0};
    double p2{0};
    double k3{0};
    double k4{0};
    double k5{0};
    double k6{0};
};

/// @brief
/// Fisheye distortion model with 4 coefficients (k1, k2, k3, k4).
struct DistortionFisheye
{
    double k1{0};
    double k2{0};
    double k3{0};
    double k4{0};
};
} // namespace retinify