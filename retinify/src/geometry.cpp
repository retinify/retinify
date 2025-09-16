// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/geometry.hpp"

#include <cmath>

namespace retinify
{
constexpr double EPS = 1e-12;
constexpr double PI = 3.14159265358979323846264338327950288;

auto Identity() noexcept -> Mat3x3d
{
    return {{{1.0, 0.0, 0.0}, //
             {0.0, 1.0, 0.0}, //
             {0.0, 0.0, 1.0}}};
}

auto Determinant(const Mat3x3d &mat) noexcept -> double
{
    return mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) - //
           mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) + //
           mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
}

auto Transpose(const Mat3x3d &mat) noexcept -> Mat3x3d
{
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
    Vec3d vecOut{};
    for (int i = 0; i < 3; ++i)
    {
        vecOut[i] = mat[i][0] * vec[0] + mat[i][1] * vec[1] + mat[i][2] * vec[2];
    }
    return vecOut;
}

auto Multiply(const Mat3x3d &mat1, const Mat3x3d &mat2) noexcept -> Mat3x3d
{
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
    return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

auto Normalize(const Vec3d &vec) noexcept -> Vec3d
{
    double n = Length(vec);
    if (n < EPS)
    {
        return {0, 0, 0};
    }
    return {vec[0] / n, vec[1] / n, vec[2] / n};
}

auto Cross(const Vec3d &vec1, const Vec3d &vec2) noexcept -> Vec3d
{
    return {vec1[1] * vec2[2] - vec1[2] * vec2[1], vec1[2] * vec2[0] - vec1[0] * vec2[2], vec1[0] * vec2[1] - vec1[1] * vec2[0]};
}

auto Exp(const Vec3d &omega) noexcept -> Mat3x3d
{
    // Robust Rodrigues: R = I + a*[w]_x + b*([w]_x)^2 with
    // a = sin(theta)/theta, b = (1-cos(theta))/theta^2; use series for small theta.
    const double wx = omega[0], wy = omega[1], wz = omega[2];
    const double theta2 = wx * wx + wy * wy + wz * wz;

    Mat3x3d R{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    double a = 0.0;
    double b = 0.0;
    if (theta2 <= EPS)
    {
        // a = 1 - t^2/6 + t^4/120..., b = 1/2 - t^2/24 + t^4/720...
        const double t2 = theta2;
        const double t4 = t2 * t2;
        a = 1.0 - t2 / 6.0 + t4 / 120.0;
        b = 0.5 - t2 / 24.0 + t4 / 720.0;
    }
    else
    {
        const double theta = std::sqrt(theta2);
        a = std::sin(theta) / theta;
        b = (1.0 - std::cos(theta)) / theta2;
    }

    // [w]_x
    const double W01 = -wz, W02 = wy;
    const double W10 = wz, W12 = -wx;
    const double W20 = -wy, W21 = wx;

    // Components reused in (w w^T - theta^2 I)
    const double wx2 = wx * wx;
    const double wy2 = wy * wy;
    const double wz2 = wz * wz;
    const double wxwy = wx * wy;
    const double wxwz = wx * wz;
    const double wywz = wy * wz;

    // R = I + a*W + b*(w w^T - theta^2 I)
    R[0][0] -= b * (wy2 + wz2);
    R[0][1] += a * W01 + b * wxwy;
    R[0][2] += a * W02 + b * wxwz;

    R[1][0] += a * W10 + b * wxwy;
    R[1][1] -= b * (wx2 + wz2);
    R[1][2] += a * W12 + b * wywz;

    R[2][0] += a * W20 + b * wxwz;
    R[2][1] += a * W21 + b * wywz;
    R[2][2] -= b * (wx2 + wy2);

    return R;
}

auto Clamp(double x, double lo, double hi) noexcept -> double
{
    return x < lo ? lo : (x > hi ? hi : x);
}

auto Log(const Mat3x3d &R) noexcept -> Vec3d
{
    // Compute angle from trace
    const double trace = R[0][0] + R[1][1] + R[2][2];
    const double cos_theta = Clamp((trace - 1.0) * 0.5, -1.0, 1.0);
    const double theta = std::acos(cos_theta);

    // Skew-symmetric part -> 2*sin(theta)*k
    const double vx = R[2][1] - R[1][2];
    const double vy = R[0][2] - R[2][0];
    const double vz = R[1][0] - R[0][1];

    // Small angle: omega ~ 1/2 * [v]
    if (theta < EPS)
    {
        return {0.5 * vx, 0.5 * vy, 0.5 * vz};
    }

    // Near-pi handling: sin(theta) ~ 0 -> use diagonal method
    if (std::fabs(PI - theta) < 1e-6)
    {
        double xx = std::max(0.0, (R[0][0] + 1.0) * 0.5);
        double yy = std::max(0.0, (R[1][1] + 1.0) * 0.5);
        double zz = std::max(0.0, (R[2][2] + 1.0) * 0.5);

        double x = std::sqrt(xx);
        double y = std::sqrt(yy);
        double z = std::sqrt(zz);

        if (x >= y && x >= z)
        {
            // x dominant
            x = (x > 0.0) ? x : 0.0;
            y = (R[0][1] + R[1][0]) / (4.0 * std::max(x, EPS));
            z = (R[0][2] + R[2][0]) / (4.0 * std::max(x, EPS));
        }
        else if (y >= x && y >= z)
        {
            double y0 = (y > 0.0) ? y : 0.0;
            x = (R[0][1] + R[1][0]) / (4.0 * std::max(y0, EPS));
            z = (R[1][2] + R[2][1]) / (4.0 * std::max(y0, EPS));
            y = y0;
        }
        else
        {
            double z0 = (z > 0.0) ? z : 0.0;
            x = (R[0][2] + R[2][0]) / (4.0 * std::max(z0, EPS));
            y = (R[1][2] + R[2][1]) / (4.0 * std::max(z0, EPS));
            z = z0;
        }

        Vec3d k = Normalize({x, y, z});
        return {k[0] * theta, k[1] * theta, k[2] * theta};
    }

    // General case
    const double s2 = vx * vx + vy * vy + vz * vz; // = (2*sin(theta))^2
    const double s = std::sqrt(std::max(0.0, s2));
    if (s < EPS)
    {
        return {0.0, 0.0, 0.0};
    }
    const double scale = theta / s; // since v/|v| = k
    return {scale * vx, scale * vy, scale * vz};
}
} // namespace retinify
