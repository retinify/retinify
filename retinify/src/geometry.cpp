// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/geometry.hpp"

#include <cmath>
#include <numbers>

namespace retinify
{
/// @brief Small epsilon constant for numerical stability.
constexpr double EPS = 1e-12;

/// @brief Mathematical constant π (pi).
constexpr double PI = 3.14159265358979323846264338327950288;

/// @brief Clamp value into [lower, upper] (constexpr).
constexpr double Clamp(double value, double lower, double upper) noexcept
{
    return value < lower ? lower : (value > upper ? upper : value);
}

auto Identity() noexcept -> Mat3x3d
{
    return {{{1.0, 0.0, 0.0}, //
             {0.0, 1.0, 0.0}, //
             {0.0, 0.0, 1.0}}};
}

auto Determinant(const Mat3x3d &mat) noexcept -> double
{
    // det(R) = r00(r11 r22 - r12 r21) - r01(r10 r22 - r12 r20) + r02(r10 r21 - r11 r20)
    return mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) - //
           mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) + //
           mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
}

auto Transpose(const Mat3x3d &mat) noexcept -> Mat3x3d
{
    // (R^T)_{ij} = R_{ji}
    Mat3x3d matOut{};
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            matOut[i][j] = mat[j][i];
        }
    }
    return matOut;
}

auto Multiply(const Mat3x3d &mat, const Vec3d &vec) noexcept -> Vec3d
{
    // y = R x
    Vec3d vecOut{};
    for (int i = 0; i < 3; ++i)
    {
        vecOut[i] = mat[i][0] * vec[0] + mat[i][1] * vec[1] + mat[i][2] * vec[2];
    }
    return vecOut;
}

auto Multiply(const Mat3x3d &mat1, const Mat3x3d &mat2) noexcept -> Mat3x3d
{
    // C = A B,  c_{ij} = Σ_k a_{ik} b_{kj}
    Mat3x3d matOut{};
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            matOut[i][j] = mat1[i][0] * mat2[0][j] + mat1[i][1] * mat2[1][j] + mat1[i][2] * mat2[2][j];
        }
    }
    return matOut;
}

auto Length(const Vec3d &vec) noexcept -> double
{
    // ||v|| = sqrt(v·v)
    return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

auto Normalize(const Vec3d &vec) noexcept -> Vec3d
{
    // v / ||v||  (return zero if degenerate)
    const double n = Length(vec);
    if (n < EPS)
    {
        return {0.0, 0.0, 0.0};
    }
    return {vec[0] / n, vec[1] / n, vec[2] / n};
}

auto Cross(const Vec3d &vec1, const Vec3d &vec2) noexcept -> Vec3d
{
    // v1 × v2
    return {vec1[1] * vec2[2] - vec1[2] * vec2[1], //
            vec1[2] * vec2[0] - vec1[0] * vec2[2], //
            vec1[0] * vec2[1] - vec1[1] * vec2[0]};
}

auto Exp(const Vec3d &omega) noexcept -> Mat3x3d
{
    // Rodrigues: R = I + coefA[ω]_x + coefB([ω]_x)^2
    // coefA = sinθ/θ, coefB = (1−cosθ)/θ^2
    // Use series expansion for small θ.
    const double wx = omega[0];
    const double wy = omega[1];
    const double wz = omega[2];
    const double thetaSquared = wx * wx + wy * wy + wz * wz;

    Mat3x3d rotation{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
    double coefA = 0.0;
    double coefB = 0.0;

    if (thetaSquared <= EPS)
    {
        // coefA = 1 − θ^2/6 + θ^4/120
        // coefB = 1/2 − θ^2/24 + θ^4/720
        const double t2 = thetaSquared;
        const double t4 = t2 * t2;
        coefA = 1.0 - t2 / 6.0 + t4 / 120.0;
        coefB = 0.5 - t2 / 24.0 + t4 / 720.0;
    }
    else
    {
        const double theta = std::sqrt(thetaSquared);
        coefA = std::sin(theta) / theta;
        coefB = (1.0 - std::cos(theta)) / thetaSquared;
    }

    // [ω]_x components
    const double w01 = -wz;
    const double w02 = wy;
    const double w10 = wz;
    const double w12 = -wx;
    const double w20 = -wy;
    const double w21 = wx;

    // Precompute ωω^T terms
    const double wxx = wx * wx;
    const double wyy = wy * wy;
    const double wzz = wz * wz;
    const double wxy = wx * wy;
    const double wxz = wx * wz;
    const double wyz = wy * wz;

    // R = I + coefA[ω]_x + coefB(ωω^T − θ^2 I)
    rotation[0][0] -= coefB * (wyy + wzz);
    rotation[0][1] += coefA * w01 + coefB * wxy;
    rotation[0][2] += coefA * w02 + coefB * wxz;

    rotation[1][0] += coefA * w10 + coefB * wxy;
    rotation[1][1] -= coefB * (wxx + wzz);
    rotation[1][2] += coefA * w12 + coefB * wyz;

    rotation[2][0] += coefA * w20 + coefB * wxz;
    rotation[2][1] += coefA * w21 + coefB * wyz;
    rotation[2][2] -= coefB * (wxx + wyy);

    return rotation;
}

auto Log(const Mat3x3d &rotation) noexcept -> Vec3d
{
    // θ from trace: cosθ = (tr(R) − 1)/2
    const double tr = rotation[0][0] + rotation[1][1] + rotation[2][2];
    const double cosTheta = Clamp((tr - 1.0) * 0.5, -1.0, 1.0);
    const double theta = std::acos(cosTheta);

    // v = (R − R^T)∨ = 2 sinθ · n  (skew vector)
    const double vx = rotation[2][1] - rotation[1][2];
    const double vy = rotation[0][2] - rotation[2][0];
    const double vz = rotation[1][0] - rotation[0][1];

    // Small angle: ω ≈ 1/2 v (since sinθ ≈ θ)
    if (theta < EPS)
    {
        return {0.5 * vx, 0.5 * vy, 0.5 * vz};
    }

    // Near π: sinθ ~ 0, use diagonal-based axis extraction.
    if (std::fabs(PI - theta) < 1e-6)
    {
        // n^2 from diagonal: n_x^2 = (R_xx + 1)/2, etc., clamped to [0,1].
        double ax = std::sqrt(std::max(0.0, (rotation[0][0] + 1.0) * 0.5));
        double ay = std::sqrt(std::max(0.0, (rotation[1][1] + 1.0) * 0.5));
        double az = std::sqrt(std::max(0.0, (rotation[2][2] + 1.0) * 0.5));

        // Recover remaining components from off-diagonals using the largest axis to stabilize.
        if (ax >= ay && ax >= az)
        {
            const double denom = 4.0 * std::max(ax, EPS);
            ay = (rotation[0][1] + rotation[1][0]) / denom;
            az = (rotation[0][2] + rotation[2][0]) / denom;
        }
        else if (ay >= ax && ay >= az)
        {
            const double denom = 4.0 * std::max(ay, EPS);
            ax = (rotation[0][1] + rotation[1][0]) / denom;
            az = (rotation[1][2] + rotation[2][1]) / denom;
        }
        else
        {
            const double denom = 4.0 * std::max(az, EPS);
            ax = (rotation[0][2] + rotation[2][0]) / denom;
            ay = (rotation[1][2] + rotation[2][1]) / denom;
        }

        // ω = θ n
        const Vec3d axis = Normalize({ax, ay, az});
        return {axis[0] * theta, axis[1] * theta, axis[2] * theta};
    }

    // General case: n = v / ||v||, ω = θ n
    const double vnormSquared = vx * vx + vy * vy + vz * vz;
    const double vnorm = std::sqrt(std::max(0.0, vnormSquared));
    if (vnorm < EPS)
    {
        return {0.0, 0.0, 0.0};
    }
    const double scale = theta / vnorm;
    return {scale * vx, scale * vy, scale * vz};
}
} // namespace retinify
