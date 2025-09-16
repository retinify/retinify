// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/geometry.hpp"

#include <cmath>

#include <gtest/gtest.h>

namespace retinify
{
constexpr double kTol = 1e-9;

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
    EXPECT_NEAR(Determinant(R), 1.0, kTol);
}

TEST(GeometryTest, MultiplyMatrixVector)
{
    const retinify::Mat3x3d mat{{{1.0, 2.0, 3.0}, {0.0, -1.0, 4.0}, {2.5, 0.5, 1.0}}};
    const retinify::Vec3d vec{2.0, -1.0, 0.5};

    const retinify::Vec3d result = retinify::Multiply(mat, vec);

    ExpectVectorNear(result, {1.5, 3.0, 5.0}, kTol);
}

TEST(GeometryTest, MultiplyWithIdentityPreservesMatrix)
{
    const retinify::Mat3x3d A{{{1.0, 2.0, -1.0}, {0.5, -0.25, 3.0}, {4.0, 1.0, 0.0}}};
    const retinify::Mat3x3d I = Identity();

    ExpectMatrixNear(retinify::Multiply(I, A), A, kTol);
    ExpectMatrixNear(retinify::Multiply(A, I), A, kTol);
}

TEST(GeometryTest, TransposeIsInvolution)
{
    const retinify::Mat3x3d mat{{{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}, {6.0, 7.0, 8.0}}};
    ExpectMatrixNear(retinify::Transpose(retinify::Transpose(mat)), mat, kTol);
}

TEST(GeometryTest, LengthAndNormalize)
{
    const retinify::Vec3d vec{3.0, 4.0, 12.0};
    EXPECT_NEAR(retinify::Length(vec), std::sqrt(169.0), kTol);

    const retinify::Vec3d unit = retinify::Normalize(vec);
    EXPECT_NEAR(retinify::Length(unit), 1.0, kTol);

    const retinify::Vec3d zero{0.0, 0.0, 0.0};
    ExpectVectorNear(retinify::Normalize(zero), zero, kTol);
}

TEST(GeometryTest, CrossProductOrthogonality)
{
    const retinify::Vec3d u{1.0, 2.0, 3.0};
    const retinify::Vec3d v{-4.0, 5.0, -6.0};
    const retinify::Vec3d cross = retinify::Cross(u, v);

    EXPECT_NEAR(cross[0] * u[0] + cross[1] * u[1] + cross[2] * u[2], 0.0, kTol);
    EXPECT_NEAR(cross[0] * v[0] + cross[1] * v[1] + cross[2] * v[2], 0.0, kTol);

    const double lenUSq = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    const double lenVSq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    const double dotUV = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    const double expectedSq = lenUSq * lenVSq - dotUV * dotUV;
    const double crossSq = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
    EXPECT_NEAR(crossSq, expectedSq, kTol);

    const retinify::Vec3d reversed = retinify::Cross(v, u);
    ExpectVectorNear(reversed, {-cross[0], -cross[1], -cross[2]}, kTol);
}

TEST(GeometryTest, ExpReturnsIdentityForZeroRotation)
{
    ExpectMatrixNear(retinify::Exp({0.0, 0.0, 0.0}), Identity(), kTol);
}

TEST(GeometryTest, ExpMatchesCardinalAxisRotations)
{
    constexpr double theta = 0.5;
    const double c = std::cos(theta);
    const double s = std::sin(theta);

    const retinify::Mat3x3d Rx = retinify::Exp({theta, 0.0, 0.0});
    ExpectMatrixNear(Rx, {{{1.0, 0.0, 0.0}, {0.0, c, -s}, {0.0, s, c}}}, kTol);

    const retinify::Mat3x3d Ry = retinify::Exp({0.0, theta, 0.0});
    ExpectMatrixNear(Ry, {{{c, 0.0, s}, {0.0, 1.0, 0.0}, {-s, 0.0, c}}}, kTol);

    const retinify::Mat3x3d Rz = retinify::Exp({0.0, 0.0, theta});
    ExpectMatrixNear(Rz, {{{c, -s, 0.0}, {s, c, 0.0}, {0.0, 0.0, 1.0}}}, kTol);
}

TEST(GeometryTest, ExpProducesProperRotation)
{
    const retinify::Vec3d omega{0.7, -0.2, 0.4};
    const retinify::Mat3x3d R = retinify::Exp(omega);
    ExpectOrthonormal(R, kTol);
}

TEST(GeometryTest, LogOfIdentityIsZero)
{
    ExpectVectorNear(retinify::Log(Identity()), {0.0, 0.0, 0.0}, kTol);
}

TEST(GeometryTest, LogRecoversRotationVector)
{
    const retinify::Vec3d omega{1e-4, -2e-4, 1.5e-4};
    const retinify::Vec3d recovered = retinify::Log(retinify::Exp(omega));
    ExpectVectorNear(recovered, omega, kTol);
}

TEST(GeometryTest, ExpLogRoundTrip)
{
    const retinify::Mat3x3d R = retinify::Exp({-0.8, 0.3, 0.1});
    const retinify::Vec3d logR = retinify::Log(R);
    const retinify::Mat3x3d reconstructed = retinify::Exp(logR);
    ExpectMatrixNear(reconstructed, R, kTol);
}

TEST(GeometryTest, LogHandlesPiRotation)
{
    const double theta = 3.14159265358979323846;
    const retinify::Mat3x3d R{{{1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, -1.0}}};
    const retinify::Vec3d omega = retinify::Log(R);

    EXPECT_NEAR(std::fabs(omega[0]), theta, kTol);
    EXPECT_NEAR(omega[1], 0.0, kTol);
    EXPECT_NEAR(omega[2], 0.0, kTol);
}

auto DistortPoint(const Intrinsics &intrinsics, const Distortion &distortion, const Point2d &idealNormalized) noexcept -> Point2d
{
    const double x = idealNormalized[0];
    const double y = idealNormalized[1];
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;

    const double radialNumerator = 1.0 + distortion.k1 * r2 + distortion.k2 * r4 + distortion.k3 * r6;
    const double radialDenominator = 1.0 + distortion.k4 * r2 + distortion.k5 * r4 + distortion.k6 * r6;
    const double radial = (std::fabs(radialDenominator) > kTol) ? (radialNumerator / radialDenominator) : 1.0;

    const double deltaX = 2.0 * distortion.p1 * x * y + distortion.p2 * (r2 + 2.0 * x * x);
    const double deltaY = distortion.p1 * (r2 + 2.0 * y * y) + 2.0 * distortion.p2 * x * y;

    const double distortedX = x * radial + deltaX;
    const double distortedY = y * radial + deltaY;

    return {distortedX * intrinsics.fx + intrinsics.cx, distortedY * intrinsics.fy + intrinsics.cy};
}

TEST(GeometryTest, UndistortPointWithFiveCoefficients)
{
    const retinify::Intrinsics intrinsics{500.0, 480.0, 320.0, 240.0, 0.0};
    const retinify::Distortion distortion{0.12, -0.05, 0.001, 0.0005, 0.03};

    const retinify::Point2d idealPixel{0.05, -0.04};
    const retinify::Point2d distortedPixel = DistortPoint(intrinsics, distortion, idealPixel);

    const retinify::Point2d undistortedPixel = retinify::UndistortPoint(intrinsics, distortion, distortedPixel);

    EXPECT_NEAR(undistortedPixel[0], idealPixel[0], kTol);
    EXPECT_NEAR(undistortedPixel[1], idealPixel[1], kTol);
}

TEST(GeometryTest, UndistortPointWithEightCoefficients)
{
    const retinify::Intrinsics intrinsics{500.0, 480.0, 320.0, 240.0, 0.0};
    const retinify::Distortion distortion{0.08, -0.03, -0.0007, 0.0004, 0.015, 0.005, -0.002, 0.001};

    const retinify::Point2d idealPixel{-0.06, 0.045};
    const retinify::Point2d distortedPixel = DistortPoint(intrinsics, distortion, idealPixel);

    const retinify::Point2d undistortedPixel = retinify::UndistortPoint(intrinsics, distortion, distortedPixel);

    EXPECT_NEAR(undistortedPixel[0], idealPixel[0], kTol);
    EXPECT_NEAR(undistortedPixel[1], idealPixel[1], kTol);
}
} // namespace retinify
