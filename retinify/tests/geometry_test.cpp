// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/geometry.hpp"

#include <cmath>

#include <gtest/gtest.h>

namespace retinify
{
constexpr double kTol = 1e-12;

void ExpectMatrixNear(const retinify::Mat3x3d &lhs, const retinify::Mat3x3d &rhs, double tol)
{
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            EXPECT_NEAR(lhs[r][c], rhs[r][c], tol) << "entry(" << r << "," << c << ")";
        }
    }
}

void ExpectVectorNear(const retinify::Vec3d &lhs, const retinify::Vec3d &rhs, double tol)
{
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_NEAR(lhs[i], rhs[i], tol) << "entry(" << i << ")";
    }
}

void ExpectOrthonormal(const retinify::Mat3x3d &R, double tol)
{
    const retinify::Mat3x3d Rt = retinify::Transpose(R);
    const retinify::Mat3x3d shouldBeIdentity = retinify::Multiply(Rt, R);
    ExpectMatrixNear(shouldBeIdentity, Identity(), tol);
    EXPECT_NEAR(Determinant(R), 1.0, 1e-10);
}

TEST(GeometryTest, MultiplyMatrixVector)
{
    const retinify::Mat3x3d mat{{{1.0, 2.0, 3.0}, {0.0, -1.0, 4.0}, {2.5, 0.5, 1.0}}};
    const retinify::Vec3d vec{2.0, -1.0, 0.5};

    const retinify::Vec3d result = retinify::Multiply(mat, vec);

    ExpectVectorNear(result, {1.5, 3.0, 5.0}, 1e-12);
}

TEST(GeometryTest, MultiplyWithIdentityPreservesMatrix)
{
    const retinify::Mat3x3d A{{{1.0, 2.0, -1.0}, {0.5, -0.25, 3.0}, {4.0, 1.0, 0.0}}};
    const retinify::Mat3x3d I = Identity();

    ExpectMatrixNear(retinify::Multiply(I, A), A, 1e-12);
    ExpectMatrixNear(retinify::Multiply(A, I), A, 1e-12);
}

TEST(GeometryTest, TransposeIsInvolution)
{
    const retinify::Mat3x3d mat{{{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}, {6.0, 7.0, 8.0}}};
    ExpectMatrixNear(retinify::Transpose(retinify::Transpose(mat)), mat, 1e-12);
}

TEST(GeometryTest, LengthAndNormalize)
{
    const retinify::Vec3d vec{3.0, 4.0, 12.0};
    EXPECT_NEAR(retinify::Length(vec), std::sqrt(169.0), 1e-12);

    const retinify::Vec3d unit = retinify::Normalize(vec);
    EXPECT_NEAR(retinify::Length(unit), 1.0, 1e-12);

    const retinify::Vec3d zero{0.0, 0.0, 0.0};
    ExpectVectorNear(retinify::Normalize(zero), zero, 1e-12);
}

TEST(GeometryTest, CrossProductOrthogonality)
{
    const retinify::Vec3d u{1.0, 2.0, 3.0};
    const retinify::Vec3d v{-4.0, 5.0, -6.0};
    const retinify::Vec3d cross = retinify::Cross(u, v);

    EXPECT_NEAR(cross[0] * u[0] + cross[1] * u[1] + cross[2] * u[2], 0.0, 1e-12);
    EXPECT_NEAR(cross[0] * v[0] + cross[1] * v[1] + cross[2] * v[2], 0.0, 1e-12);

    const double lenUSq = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    const double lenVSq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    const double dotUV = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    const double expectedSq = lenUSq * lenVSq - dotUV * dotUV;
    const double crossSq = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
    EXPECT_NEAR(crossSq, expectedSq, 1e-12);

    const retinify::Vec3d reversed = retinify::Cross(v, u);
    ExpectVectorNear(reversed, {-cross[0], -cross[1], -cross[2]}, 1e-12);
}

TEST(GeometryTest, ExpReturnsIdentityForZeroRotation)
{
    ExpectMatrixNear(retinify::Exp({0.0, 0.0, 0.0}), Identity(), 1e-15);
}

TEST(GeometryTest, ExpMatchesCardinalAxisRotations)
{
    constexpr double theta = 0.5;
    const double c = std::cos(theta);
    const double s = std::sin(theta);

    const retinify::Mat3x3d Rx = retinify::Exp({theta, 0.0, 0.0});
    ExpectMatrixNear(Rx, {{{1.0, 0.0, 0.0}, {0.0, c, -s}, {0.0, s, c}}}, 1e-12);

    const retinify::Mat3x3d Ry = retinify::Exp({0.0, theta, 0.0});
    ExpectMatrixNear(Ry, {{{c, 0.0, s}, {0.0, 1.0, 0.0}, {-s, 0.0, c}}}, 1e-12);

    const retinify::Mat3x3d Rz = retinify::Exp({0.0, 0.0, theta});
    ExpectMatrixNear(Rz, {{{c, -s, 0.0}, {s, c, 0.0}, {0.0, 0.0, 1.0}}}, 1e-12);
}

TEST(GeometryTest, ExpProducesProperRotation)
{
    const retinify::Vec3d omega{0.7, -0.2, 0.4};
    const retinify::Mat3x3d R = retinify::Exp(omega);
    ExpectOrthonormal(R, 1e-12);
}

TEST(GeometryTest, LogOfIdentityIsZero)
{
    ExpectVectorNear(retinify::Log(Identity()), {0.0, 0.0, 0.0}, kTol);
}

TEST(GeometryTest, LogRecoversRotationVector)
{
    const retinify::Vec3d omega{1e-4, -2e-4, 1.5e-4};
    const retinify::Vec3d recovered = retinify::Log(retinify::Exp(omega));
    ExpectVectorNear(recovered, omega, 1e-10);
}

TEST(GeometryTest, ExpLogRoundTrip)
{
    const retinify::Mat3x3d R = retinify::Exp({-0.8, 0.3, 0.1});
    const retinify::Vec3d logR = retinify::Log(R);
    const retinify::Mat3x3d reconstructed = retinify::Exp(logR);
    ExpectMatrixNear(reconstructed, R, 1e-10);
}

TEST(GeometryTest, LogHandlesPiRotation)
{
    const double theta = 3.14159265358979323846;
    const retinify::Mat3x3d R{{{1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, -1.0}}};
    const retinify::Vec3d omega = retinify::Log(R);

    EXPECT_NEAR(std::fabs(omega[0]), theta, 1e-9);
    EXPECT_NEAR(omega[1], 0.0, 1e-9);
    EXPECT_NEAR(omega[2], 0.0, 1e-9);
}
} // namespace retinify