// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "attributes.hpp"

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
/// Create a 3x3 identity matrix.
/// @return
/// 3x3 identity matrix.
RETINIFY_API auto Identity() noexcept -> Mat3x3d;

/// @brief
/// Compute the determinant of a 3x3 matrix.
/// @param mat
/// 3x3 matrix.
/// @return
/// Determinant value.
RETINIFY_API auto Determinant(const Mat3x3d &mat) noexcept -> double;

/// @brief
/// Transpose a 3x3 matrix.
/// @param mat
/// 3x3 matrix.
/// @return
/// Transposed 3x3 matrix.
RETINIFY_API auto Transpose(const Mat3x3d &mat) noexcept -> Mat3x3d;

/// @brief
/// Multiply two 3x3 matrices.
/// @param mat1
/// First 3x3 matrix.
/// @param mat2
/// Second 3x3 matrix.
/// @return
/// 3x3 matrix.
RETINIFY_API auto Multiply(const Mat3x3d &mat1, const Mat3x3d &mat2) noexcept -> Mat3x3d;

/// @brief
/// Multiply a 3x3 matrix and a 3D vector.
/// @param mat
/// 3x3 matrix.
/// @param vec
/// 3D vector.
/// @return
/// 3D vector.
RETINIFY_API auto Multiply(const Mat3x3d &mat, const Vec3d &vec) noexcept -> Vec3d;

/// @brief
/// Compute the length (magnitude) of a 3D vector.
/// @param vec
/// 3D vector.
/// @return
/// Length value.
RETINIFY_API auto Length(const Vec3d &vec) noexcept -> double;

/// @brief
/// Normalize a 3D vector to unit length.
/// @param vec
/// 3D vector.
/// @return
/// Normalized 3D vector.
RETINIFY_API auto Normalize(const Vec3d &vec) noexcept -> Vec3d;

/// @brief
/// Compute the cross product of two 3D vectors.
/// @param vec1
/// First 3D vector.
/// @param vec2
/// Second 3D vector.
/// @return
/// 3D vector.
RETINIFY_API auto Cross(const Vec3d &vec1, const Vec3d &vec2) noexcept -> Vec3d;

/// @brief
/// Compute the matrix exponential of a 3D rotation vector.
/// @param omega
/// 3D rotation vector.
/// @return
/// 3x3 rotation matrix.
RETINIFY_API auto Exp(const Vec3d &omega) noexcept -> Mat3x3d;

/// @brief
/// Compute the matrix logarithm of a 3x3 rotation matrix.
/// @param R
/// 3x3 rotation matrix.
/// @return
/// 3D rotation vector.
RETINIFY_API auto Log(const Mat3x3d &R) noexcept -> Vec3d;

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
