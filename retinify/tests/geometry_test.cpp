// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/geometry.hpp"

#include "mat.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace retinify
{
namespace
{
constexpr double kTolLoose = 1e-5;
constexpr double kTolStrict = 1e-12;

void ExpectMatrixNear(const retinify::Mat3x3d &lhs, const retinify::Mat3x3d &rhs, double tol)
{
    for (int rowIndex = 0; rowIndex < 3; ++rowIndex)
    {
        for (int columnIndex = 0; columnIndex < 3; ++columnIndex)
        {
            EXPECT_NEAR(lhs[rowIndex][columnIndex], rhs[rowIndex][columnIndex], tol) << "entry(" << rowIndex << "," << columnIndex << ")";
        }
    }
}

void ExpectVectorNear(const retinify::Vec3d &lhs, const retinify::Vec3d &rhs, double tol)
{
    for (int componentIndex = 0; componentIndex < 3; ++componentIndex)
    {
        EXPECT_NEAR(lhs[componentIndex], rhs[componentIndex], tol) << "entry(" << componentIndex << ")";
    }
}

void ExpectOrthonormal(const retinify::Mat3x3d &rotationMatrix, double tol)
{
    const retinify::Mat3x3d transposedMatrix = retinify::Transpose(rotationMatrix);
    const retinify::Mat3x3d shouldBeIdentity = retinify::Multiply(transposedMatrix, rotationMatrix);
    ExpectMatrixNear(shouldBeIdentity, Identity(), tol);
    EXPECT_NEAR(Determinant(rotationMatrix), 1.0, tol);
}

void ExpectMatrix34Near(const retinify::Mat3x4d &lhs, const retinify::Mat3x4d &rhs, double tol)
{
    for (int rowIndex = 0; rowIndex < 3; ++rowIndex)
    {
        for (int columnIndex = 0; columnIndex < 4; ++columnIndex)
        {
            EXPECT_NEAR(lhs[rowIndex][columnIndex], rhs[rowIndex][columnIndex], tol) << "entry(" << rowIndex << "," << columnIndex << ")";
        }
    }
}

void ExpectMatrix44Near(const retinify::Mat4x4d &lhs, const retinify::Mat4x4d &rhs, double tol)
{
    for (int rowIndex = 0; rowIndex < 4; ++rowIndex)
    {
        for (int columnIndex = 0; columnIndex < 4; ++columnIndex)
        {
            EXPECT_NEAR(lhs[rowIndex][columnIndex], rhs[rowIndex][columnIndex], tol) << "entry(" << rowIndex << "," << columnIndex << ")";
        }
    }
}
} // namespace

TEST(GeometryTest, MultiplyMatrixVector)
{
    const retinify::Mat3x3d mat{{{1.0, 2.0, 3.0}, {0.0, -1.0, 4.0}, {2.5, 0.5, 1.0}}};
    const retinify::Vec3d vec{2.0, -1.0, 0.5};

    const retinify::Vec3d result = retinify::Multiply(mat, vec);

    ExpectVectorNear(result, {1.5, 3.0, 5.0}, kTolStrict);
}

TEST(GeometryTest, MultiplyWithIdentityPreservesMatrix)
{
    const retinify::Mat3x3d matrixA{{{1.0, 2.0, -1.0}, {0.5, -0.25, 3.0}, {4.0, 1.0, 0.0}}};
    const retinify::Mat3x3d identityMatrix = Identity();

    ExpectMatrixNear(retinify::Multiply(identityMatrix, matrixA), matrixA, kTolStrict);
    ExpectMatrixNear(retinify::Multiply(matrixA, identityMatrix), matrixA, kTolStrict);
}

TEST(GeometryTest, TransposeIsInvolution)
{
    const retinify::Mat3x3d mat{{{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}, {6.0, 7.0, 8.0}}};
    ExpectMatrixNear(retinify::Transpose(retinify::Transpose(mat)), mat, kTolStrict);
}

TEST(GeometryTest, LengthAndNormalize)
{
    const retinify::Vec3d vec{3.0, 4.0, 12.0};
    EXPECT_NEAR(retinify::Length(vec), std::sqrt(169.0), kTolStrict);

    const retinify::Vec3d unit = retinify::Normalize(vec);
    EXPECT_NEAR(retinify::Length(unit), 1.0, kTolStrict);

    const retinify::Vec3d zero{0.0, 0.0, 0.0};
    ExpectVectorNear(retinify::Normalize(zero), zero, kTolStrict);
}

TEST(GeometryTest, CrossProductOrthogonality)
{
    const retinify::Vec3d vectorU{1.0, 2.0, 3.0};
    const retinify::Vec3d vectorV{-4.0, 5.0, -6.0};
    const retinify::Vec3d cross = retinify::Cross(vectorU, vectorV);

    EXPECT_NEAR(cross[0] * vectorU[0] + cross[1] * vectorU[1] + cross[2] * vectorU[2], 0.0, kTolStrict);
    EXPECT_NEAR(cross[0] * vectorV[0] + cross[1] * vectorV[1] + cross[2] * vectorV[2], 0.0, kTolStrict);

    const double lenUSq = vectorU[0] * vectorU[0] + vectorU[1] * vectorU[1] + vectorU[2] * vectorU[2];
    const double lenVSq = vectorV[0] * vectorV[0] + vectorV[1] * vectorV[1] + vectorV[2] * vectorV[2];
    const double dotUV = vectorU[0] * vectorV[0] + vectorU[1] * vectorV[1] + vectorU[2] * vectorV[2];
    const double expectedSq = lenUSq * lenVSq - dotUV * dotUV;
    const double crossSq = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
    EXPECT_NEAR(crossSq, expectedSq, kTolStrict);

    const retinify::Vec3d reversed = retinify::Cross(vectorV, vectorU);
    ExpectVectorNear(reversed, {-cross[0], -cross[1], -cross[2]}, kTolStrict);
}

TEST(GeometryTest, ExpReturnsIdentityForZeroRotation)
{
    ExpectMatrixNear(retinify::Exp({0.0, 0.0, 0.0}), Identity(), kTolStrict);
}

TEST(GeometryTest, ExpMatchesCardinalAxisRotations)
{
    constexpr double theta = 0.5;
    const double cosTheta = std::cos(theta);
    const double sinTheta = std::sin(theta);

    const retinify::Mat3x3d Rx = retinify::Exp({theta, 0.0, 0.0});
    ExpectMatrixNear(Rx, {{{1.0, 0.0, 0.0}, {0.0, cosTheta, -sinTheta}, {0.0, sinTheta, cosTheta}}}, kTolStrict);

    const retinify::Mat3x3d Ry = retinify::Exp({0.0, theta, 0.0});
    ExpectMatrixNear(Ry, {{{cosTheta, 0.0, sinTheta}, {0.0, 1.0, 0.0}, {-sinTheta, 0.0, cosTheta}}}, kTolStrict);

    const retinify::Mat3x3d Rz = retinify::Exp({0.0, 0.0, theta});
    ExpectMatrixNear(Rz, {{{cosTheta, -sinTheta, 0.0}, {sinTheta, cosTheta, 0.0}, {0.0, 0.0, 1.0}}}, kTolStrict);
}

TEST(GeometryTest, ExpProducesProperRotation)
{
    const retinify::Vec3d omega{0.7, -0.2, 0.4};
    const retinify::Mat3x3d rotationMatrix = retinify::Exp(omega);
    ExpectOrthonormal(rotationMatrix, kTolStrict);
}

TEST(GeometryTest, LogOfIdentityIsZero)
{
    ExpectVectorNear(retinify::Log(Identity()), {0.0, 0.0, 0.0}, kTolStrict);
}

TEST(GeometryTest, LogRecoversRotationVector)
{
    const retinify::Vec3d omega{1e-4, -2e-4, 1.5e-4};
    const retinify::Vec3d recovered = retinify::Log(retinify::Exp(omega));
    ExpectVectorNear(recovered, omega, kTolStrict);
}

TEST(GeometryTest, ExpLogRoundTrip)
{
    const retinify::Mat3x3d rotationMatrix = retinify::Exp({-0.8, 0.3, 0.1});
    const retinify::Vec3d logR = retinify::Log(rotationMatrix);
    const retinify::Mat3x3d reconstructed = retinify::Exp(logR);
    ExpectMatrixNear(reconstructed, rotationMatrix, kTolStrict);
}

TEST(GeometryTest, LogHandlesPiRotation)
{
    const double theta = 3.14159265358979323846;
    const retinify::Mat3x3d rotationMatrix{{{1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, -1.0}}};
    const retinify::Vec3d omega = retinify::Log(rotationMatrix);

    EXPECT_NEAR(std::fabs(omega[0]), theta, kTolStrict);
    EXPECT_NEAR(omega[1], 0.0, kTolStrict);
    EXPECT_NEAR(omega[2], 0.0, kTolStrict);
}

auto DistortPoint(const Intrinsics &intrinsics, const Distortion &distortion, const Point2d &pixel) noexcept -> Point2d
{
    const double pixelX = pixel[0];
    const double pixelY = pixel[1];
    const double r2 = pixelX * pixelX + pixelY * pixelY;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;

    const double radialNumerator = 1.0 + distortion.k1 * r2 + distortion.k2 * r4 + distortion.k3 * r6;
    const double radialDenominator = 1.0 + distortion.k4 * r2 + distortion.k5 * r4 + distortion.k6 * r6;
    const double radial = (std::fabs(radialDenominator) > kTolStrict) ? (radialNumerator / radialDenominator) : 1.0;

    const double deltaX = 2.0 * distortion.p1 * pixelX * pixelY + distortion.p2 * (r2 + 2.0 * pixelX * pixelX);
    const double deltaY = distortion.p1 * (r2 + 2.0 * pixelY * pixelY) + 2.0 * distortion.p2 * pixelX * pixelY;

    const double distortedX = pixelX * radial + deltaX;
    const double distortedY = pixelY * radial + deltaY;

    return {distortedX * intrinsics.fx + intrinsics.cx, distortedY * intrinsics.fy + intrinsics.cy};
}

TEST(GeometryTest, UndistortPointWithFiveCoefficients)
{
    const retinify::Intrinsics intrinsics{500.0, 480.0, 320.0, 240.0, 0.0};
    const retinify::Distortion distortion{0.12, -0.05, 0.001, 0.0005, 0.03};

    const retinify::Point2d idealPixel{0.05, -0.04};
    const retinify::Point2d distortedPixel = DistortPoint(intrinsics, distortion, idealPixel);

    const retinify::Point2d undistortedPixel = retinify::UndistortPoint(intrinsics, distortion, distortedPixel);

    EXPECT_NEAR(undistortedPixel[0], idealPixel[0], kTolStrict);
    EXPECT_NEAR(undistortedPixel[1], idealPixel[1], kTolStrict);
}

TEST(GeometryTest, UndistortPointWithEightCoefficients)
{
    const retinify::Intrinsics intrinsics{500.0, 480.0, 320.0, 240.0, 0.0};
    const retinify::Distortion distortion{0.08, -0.03, -0.0007, 0.0004, 0.015, 0.005, -0.002, 0.001};

    const retinify::Point2d idealPixel{-0.06, 0.045};
    const retinify::Point2d distortedPixel = DistortPoint(intrinsics, distortion, idealPixel);

    const retinify::Point2d undistortedPixel = retinify::UndistortPoint(intrinsics, distortion, distortedPixel);

    EXPECT_NEAR(undistortedPixel[0], idealPixel[0], kTolStrict);
    EXPECT_NEAR(undistortedPixel[1], idealPixel[1], kTolStrict);
}

TEST(GeometryTest, StereoRectifyIdealRig)
{
    const retinify::Intrinsics primaryIntrinsics{500.0, 500.0, 320.0, 240.0, 0.0};
    const retinify::Intrinsics secondaryIntrinsics{500.0, 500.0, 320.0, 240.0, 0.0};
    const retinify::Distortion noDistortion{};

    const retinify::Mat3x3d rotationMatrix = Identity();
    const retinify::Vec3d translationVector{0.1, 0.0, 0.0};

    retinify::Mat3x3d rectifiedRotationFirst{};
    retinify::Mat3x3d rectifiedRotationSecond{};
    retinify::Mat3x4d projectionFirst{};
    retinify::Mat3x4d projectionSecond{};
    retinify::Mat4x4d reprojectionMatrix{};

    StereoRectify(primaryIntrinsics, noDistortion, secondaryIntrinsics, noDistortion, rotationMatrix, translationVector, 640, 480, rectifiedRotationFirst, rectifiedRotationSecond, projectionFirst, projectionSecond, reprojectionMatrix);

    ExpectMatrixNear(rectifiedRotationFirst, Identity(), kTolStrict);
    ExpectMatrixNear(rectifiedRotationSecond, Identity(), kTolStrict);

    const double expectedFocal = 500.0;
    const double expectedCx = 320.0;
    const double expectedCy = 240.0;
    retinify::Mat3x4d expectedP1{};
    expectedP1[0] = {expectedFocal, 0.0, expectedCx, 0.0};
    expectedP1[1] = {0.0, expectedFocal, expectedCy, 0.0};
    expectedP1[2] = {0.0, 0.0, 1.0, 0.0};
    ExpectMatrix34Near(projectionFirst, expectedP1, kTolStrict);

    retinify::Mat3x4d expectedP2 = expectedP1;
    expectedP2[0][3] = expectedFocal * translationVector[0];
    ExpectMatrix34Near(projectionSecond, expectedP2, kTolStrict);

    retinify::Mat4x4d expectedQ{};
    expectedQ[0][0] = 1.0;
    expectedQ[1][1] = 1.0;
    expectedQ[0][3] = -expectedCx;
    expectedQ[1][3] = -expectedCy;
    expectedQ[2][3] = expectedFocal;
    expectedQ[3][2] = -1.0 / translationVector[0];
    ExpectMatrix44Near(reprojectionMatrix, expectedQ, kTolStrict);
}

TEST(GeometryTest, StereoRectifyHorizontalBaseline)
{
    const retinify::Intrinsics K1{620.0, 590.0, 310.0, 245.0, 0.0};
    const retinify::Intrinsics K2{600.0, 575.0, 305.0, 250.0, 0.0};
    const retinify::Distortion D1{0.01, -0.005, 0.0005, -0.0003, 0.001, 0.0002, -1e-4, 5e-5};
    const retinify::Distortion D2{-0.008, 0.004, -0.0004, 8e-4, -0.0009, 0.0003, 2e-4, -6e-5};

    const retinify::Mat3x3d rotationMatrix = retinify::Exp({0.1, -0.05, 0.07});
    const retinify::Vec3d translationVector{0.12, 0.03, -0.02};

    retinify::Mat3x3d rectifiedRotationFirst{};
    retinify::Mat3x3d rectifiedRotationSecond{};
    retinify::Mat3x4d projectionFirst{};
    retinify::Mat3x4d projectionSecond{};
    retinify::Mat4x4d reprojectionMatrix{};

    StereoRectify(K1, D1, K2, D2, rotationMatrix, translationVector, 800, 600, rectifiedRotationFirst, rectifiedRotationSecond, projectionFirst, projectionSecond, reprojectionMatrix);

    ExpectOrthonormal(rectifiedRotationFirst, kTolStrict);
    ExpectOrthonormal(rectifiedRotationSecond, kTolStrict);

    const double offsetX = projectionSecond[0][3];
    const double offsetY = projectionSecond[1][3];
    EXPECT_GT(std::fabs(offsetX), std::fabs(offsetY));
    EXPECT_NEAR(offsetY, 0.0, kTolStrict);
    const double expectedFocal = (K1.fy + K2.fy) * 0.5;

    EXPECT_NEAR(projectionFirst[0][0], expectedFocal, kTolStrict);
    EXPECT_NEAR(projectionFirst[1][1], expectedFocal, kTolStrict);
    EXPECT_NEAR(projectionSecond[0][0], expectedFocal, kTolStrict);
    EXPECT_NEAR(projectionSecond[1][1], expectedFocal, kTolStrict);

    EXPECT_NEAR(projectionFirst[0][2], projectionSecond[0][2], kTolStrict);
    EXPECT_NEAR(projectionFirst[1][2], projectionSecond[1][2], kTolStrict);
    EXPECT_NEAR(projectionFirst[0][2], -reprojectionMatrix[0][3], kTolStrict);
    EXPECT_NEAR(projectionFirst[1][2], -reprojectionMatrix[1][3], kTolStrict);
    EXPECT_NEAR(reprojectionMatrix[2][3], expectedFocal, kTolStrict);

    const double translationOffset = offsetX;
    ASSERT_GT(std::fabs(translationOffset), kTolStrict);
    const double baselineComponent = translationOffset / expectedFocal;
    const retinify::Vec3d rectifiedTranslation = retinify::Multiply(rectifiedRotationSecond, translationVector);
    EXPECT_NEAR(rectifiedTranslation[0], baselineComponent, kTolStrict);
    EXPECT_NEAR(rectifiedTranslation[1], 0.0, kTolStrict);
    EXPECT_NEAR(rectifiedTranslation[2], 0.0, kTolStrict);

    EXPECT_NEAR(reprojectionMatrix[3][2], -1.0 / baselineComponent, kTolStrict);
}

TEST(GeometryTest, StereoRectifyVerticalBaseline)
{
    const retinify::Intrinsics K1{580.0, 615.0, 315.0, 255.0, 0.0};
    const retinify::Intrinsics K2{590.0, 605.0, 320.0, 248.0, 0.0};
    const retinify::Distortion D1{-0.006, 0.003, -0.0002, 0.0004, -0.0007, 1e-4, -5e-5, 2e-5};
    const retinify::Distortion D2{0.005, -0.002, 3e-4, -4e-4, 8e-4, -2e-4, 7e-5, -3e-5};

    const retinify::Mat3x3d rotationMatrix = retinify::Exp({-0.04, 0.02, 0.05});
    const retinify::Vec3d translationVector{0.015, 0.2, -0.025};

    retinify::Mat3x3d rectifiedRotationFirst{};
    retinify::Mat3x3d rectifiedRotationSecond{};
    retinify::Mat3x4d projectionFirst{};
    retinify::Mat3x4d projectionSecond{};
    retinify::Mat4x4d reprojectionMatrix{};

    StereoRectify(K1, D1, K2, D2, rotationMatrix, translationVector, 1024, 768, rectifiedRotationFirst, rectifiedRotationSecond, projectionFirst, projectionSecond, reprojectionMatrix);

    ExpectOrthonormal(rectifiedRotationFirst, kTolStrict);
    ExpectOrthonormal(rectifiedRotationSecond, kTolStrict);

    const double offsetX = projectionSecond[0][3];
    const double offsetY = projectionSecond[1][3];
    EXPECT_GT(std::fabs(offsetY), std::fabs(offsetX));
    EXPECT_NEAR(offsetX, 0.0, kTolStrict);
    const double expectedFocal = (K1.fx + K2.fx) * 0.5;

    EXPECT_NEAR(projectionFirst[0][0], expectedFocal, kTolStrict);
    EXPECT_NEAR(projectionFirst[1][1], expectedFocal, kTolStrict);
    EXPECT_NEAR(projectionSecond[0][0], expectedFocal, kTolStrict);
    EXPECT_NEAR(projectionSecond[1][1], expectedFocal, kTolStrict);

    EXPECT_NEAR(projectionFirst[0][2], projectionSecond[0][2], kTolStrict);
    EXPECT_NEAR(projectionFirst[1][2], projectionSecond[1][2], kTolStrict);
    EXPECT_NEAR(projectionFirst[0][2], -reprojectionMatrix[0][3], kTolStrict);
    EXPECT_NEAR(projectionFirst[1][2], -reprojectionMatrix[1][3], kTolStrict);
    EXPECT_NEAR(reprojectionMatrix[2][3], expectedFocal, kTolStrict);

    const double translationOffset = offsetY;
    ASSERT_GT(std::fabs(translationOffset), kTolStrict);
    const double baselineComponent = translationOffset / expectedFocal;
    const retinify::Vec3d rectifiedTranslation = retinify::Multiply(rectifiedRotationSecond, translationVector);
    EXPECT_NEAR(rectifiedTranslation[1], baselineComponent, kTolStrict);
    EXPECT_NEAR(rectifiedTranslation[0], 0.0, kTolStrict);
    EXPECT_NEAR(rectifiedTranslation[2], 0.0, kTolStrict);

    EXPECT_NEAR(reprojectionMatrix[3][2], -1.0 / baselineComponent, kTolStrict);
}

TEST(GeometryTest, InitUndistortRectifyMapIdentity)
{
    const Intrinsics intrinsics{1.0, 1.0, 0.0, 0.0, 0.0};
    const Distortion distortion{};
    const Mat3x3d rectificationRotation = Identity();

    Mat3x4d projection{};
    projection[0][0] = 1.0;
    projection[1][1] = 1.0;
    projection[2][2] = 1.0;

    constexpr int kWidth = 3;
    constexpr int kHeight = 2;

    Mat mapX;
    Mat mapY;
    const Status allocStatusX = mapX.Allocate(kHeight, kWidth, 1, sizeof(float), MatLocation::HOST);
    ASSERT_TRUE(allocStatusX.IsOK());
    const Status allocStatusY = mapY.Allocate(kHeight, kWidth, 1, sizeof(float), MatLocation::HOST);
    ASSERT_TRUE(allocStatusY.IsOK());

    InitUndistortRectifyMap(intrinsics, distortion, rectificationRotation, projection, kWidth, kHeight, //
                            static_cast<float *>(mapX.Data()), mapX.Stride(),                           //
                            static_cast<float *>(mapY.Data()), mapY.Stride());

    const void *mapXVoid = mapX.Data();
    const void *mapYVoid = mapY.Data();
    const auto *mapXBytes = static_cast<const std::byte *>(mapXVoid);
    const auto *mapYBytes = static_cast<const std::byte *>(mapYVoid);
    for (int v = 0; v < kHeight; ++v)
    {
        const std::size_t offsetX = static_cast<std::size_t>(v) * mapX.Stride();
        const std::size_t offsetY = static_cast<std::size_t>(v) * mapY.Stride();
        const auto *mapXRow = static_cast<const float *>(static_cast<const void *>(mapXBytes + offsetX));
        const auto *mapYRow = static_cast<const float *>(static_cast<const void *>(mapYBytes + offsetY));
        for (int u = 0; u < kWidth; ++u)
        {
            EXPECT_NEAR(mapXRow[u], static_cast<float>(u), kTolStrict);
            EXPECT_NEAR(mapYRow[u], static_cast<float>(v), kTolStrict);
        }
    }
}

TEST(GeometryTest, InitUndistortRectifyMapRotatedCamera)
{
    const Intrinsics intrinsics{4.0, 5.0, 3.0, 2.0, 0.1};
    const Distortion distortion{};
    const Mat3x3d rectificationRotation{{{0.0, -1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}}};

    Mat3x4d projection{};
    projection[0] = {2.0, 0.0, 1.5, 0.0};
    projection[1] = {0.0, 2.0, 0.5, 0.0};
    projection[2] = {0.0, 0.0, 1.0, 0.0};

    constexpr int kWidth = 4;
    constexpr int kHeight = 3;

    Mat mapX;
    Mat mapY;
    const Status allocStatusX = mapX.Allocate(kHeight, kWidth, 1, sizeof(float), MatLocation::HOST);
    ASSERT_TRUE(allocStatusX.IsOK());
    const Status allocStatusY = mapY.Allocate(kHeight, kWidth, 1, sizeof(float), MatLocation::HOST);
    ASSERT_TRUE(allocStatusY.IsOK());

    InitUndistortRectifyMap(intrinsics, distortion, rectificationRotation, projection, kWidth, kHeight, //
                            static_cast<float *>(mapX.Data()), mapX.Stride(),                           //
                            static_cast<float *>(mapY.Data()), mapY.Stride());

    const void *mapXVoid = mapX.Data();
    const void *mapYVoid = mapY.Data();
    const auto *mapXBytes = static_cast<const std::byte *>(mapXVoid);
    const auto *mapYBytes = static_cast<const std::byte *>(mapYVoid);
    for (int v = 0; v < kHeight; ++v)
    {
        const std::size_t offsetX = static_cast<std::size_t>(v) * mapX.Stride();
        const std::size_t offsetY = static_cast<std::size_t>(v) * mapY.Stride();
        const auto *mapXRow = static_cast<const float *>(static_cast<const void *>(mapXBytes + offsetX));
        const auto *mapYRow = static_cast<const float *>(static_cast<const void *>(mapYBytes + offsetY));
        const double rectifiedY = (static_cast<double>(v) - projection[1][2]) / projection[1][1];
        for (int u = 0; u < kWidth; ++u)
        {
            const double rectifiedX = (static_cast<double>(u) - projection[0][2]) / projection[0][0];

            const double normalizedX = rectifiedY;
            const double normalizedY = -rectifiedX;

            const double expectedX = intrinsics.fx * normalizedX + intrinsics.skew * normalizedY + intrinsics.cx;
            const double expectedY = intrinsics.fy * normalizedY + intrinsics.cy;

            EXPECT_NEAR(mapXRow[u], static_cast<float>(expectedX), kTolStrict);
            EXPECT_NEAR(mapYRow[u], static_cast<float>(expectedY), kTolStrict);
        }
    }
}

TEST(GeometryTest, InitUndistortRectifyMapAppliesDistortion)
{
    const Intrinsics intrinsics{500.0, 520.0, 320.0, 240.0, 0.0};
    const Distortion distortion{0.05, -0.01, 0.001, -0.0004, 0.0005, -0.0002, 0.0001, -5e-5};
    const Mat3x3d rectificationRotation = Identity();

    Mat3x4d projection{};
    projection[0] = {450.0, 0.0, 300.0, 0.0};
    projection[1] = {0.0, 460.0, 200.0, 0.0};
    projection[2] = {0.0, 0.0, 1.0, 0.0};

    constexpr int kWidth = 2;
    constexpr int kHeight = 2;

    Mat mapX;
    Mat mapY;
    const Status allocStatusX = mapX.Allocate(kHeight, kWidth, 1, sizeof(float), MatLocation::HOST);
    ASSERT_TRUE(allocStatusX.IsOK());
    const Status allocStatusY = mapY.Allocate(kHeight, kWidth, 1, sizeof(float), MatLocation::HOST);
    ASSERT_TRUE(allocStatusY.IsOK());

    InitUndistortRectifyMap(intrinsics, distortion, rectificationRotation, projection, kWidth, kHeight, //
                            static_cast<float *>(mapX.Data()), mapX.Stride(),                           //
                            static_cast<float *>(mapY.Data()), mapY.Stride());

    const void *mapXVoid = mapX.Data();
    const void *mapYVoid = mapY.Data();
    const auto *mapXBytes = static_cast<const std::byte *>(mapXVoid);
    const auto *mapYBytes = static_cast<const std::byte *>(mapYVoid);
    for (int v = 0; v < kHeight; ++v)
    {
        const std::size_t offsetX = static_cast<std::size_t>(v) * mapX.Stride();
        const std::size_t offsetY = static_cast<std::size_t>(v) * mapY.Stride();
        const auto *mapXRow = static_cast<const float *>(static_cast<const void *>(mapXBytes + offsetX));
        const auto *mapYRow = static_cast<const float *>(static_cast<const void *>(mapYBytes + offsetY));
        for (int u = 0; u < kWidth; ++u)
        {
            const double rectifiedX = (static_cast<double>(u) - projection[0][2]) / projection[0][0];
            const double rectifiedY = (static_cast<double>(v) - projection[1][2]) / projection[1][1];

            const Point2d idealPixel{rectifiedX, rectifiedY};
            const Point2d distortedPixel = DistortPoint(intrinsics, distortion, idealPixel);

            EXPECT_NEAR(mapXRow[u], static_cast<float>(distortedPixel[0]), kTolStrict);
            EXPECT_NEAR(mapYRow[u], static_cast<float>(distortedPixel[1]), kTolStrict);
        }
    }
}

namespace
{
auto ToCvCameraMatrix(const Intrinsics &intrinsics) -> cv::Mat
{
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 0) = intrinsics.fx;
    cameraMatrix.at<double>(0, 1) = intrinsics.skew;
    cameraMatrix.at<double>(0, 2) = intrinsics.cx;
    cameraMatrix.at<double>(1, 1) = intrinsics.fy;
    cameraMatrix.at<double>(1, 2) = intrinsics.cy;
    return cameraMatrix;
}

auto ToCvDistCoeffs(const Distortion &distortion) -> cv::Mat
{
    cv::Mat distCoeffs = cv::Mat::zeros(1, 8, CV_64F);
    distCoeffs.at<double>(0, 0) = distortion.k1;
    distCoeffs.at<double>(0, 1) = distortion.k2;
    distCoeffs.at<double>(0, 2) = distortion.p1;
    distCoeffs.at<double>(0, 3) = distortion.p2;
    distCoeffs.at<double>(0, 4) = distortion.k3;
    distCoeffs.at<double>(0, 5) = distortion.k4;
    distCoeffs.at<double>(0, 6) = distortion.k5;
    distCoeffs.at<double>(0, 7) = distortion.k6;
    return distCoeffs;
}

auto ToCvRotation(const Mat3x3d &rotation) -> cv::Mat
{
    cv::Mat cvRotation(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            cvRotation.at<double>(r, c) = rotation[r][c];
        }
    }
    return cvRotation;
}

auto ToCvTranslation(const Vec3d &translation) -> cv::Mat
{
    cv::Mat cvTranslation(3, 1, CV_64F);
    for (int r = 0; r < 3; ++r)
    {
        cvTranslation.at<double>(r, 0) = translation[r];
    }
    return cvTranslation;
}

auto ToMat33d(const cv::Mat &cvMat) -> Mat3x3d
{
    Mat3x3d result{};
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            result[r][c] = cvMat.at<double>(r, c);
        }
    }
    return result;
}

auto ToMat34d(const cv::Mat &cvMat) -> Mat3x4d
{
    Mat3x4d result{};
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            result[r][c] = cvMat.at<double>(r, c);
        }
    }
    return result;
}

auto ToMat44d(const cv::Mat &cvMat) -> Mat4x4d
{
    Mat4x4d result{};
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            result[r][c] = cvMat.at<double>(r, c);
        }
    }
    return result;
}
} // namespace

TEST(GeometryTest, StereoRectifyMatchesOpenCV)
{
    const Intrinsics intrinsics1{615.0, 605.0, 321.5, 242.0, 0.0};
    const Intrinsics intrinsics2{590.0, 600.0, 318.0, 239.5, 0.0};
    const Distortion distortion1{0.011, -0.004, 5e-4, -3e-4, 7e-4, -2e-4, 1e-4, -5e-5};
    const Distortion distortion2{-0.009, 0.0035, -4e-4, 2.5e-4, -6e-4, 1.5e-4, -8e-5, 4e-5};

    const Mat3x3d rotation = Exp({0.08, -0.06, 0.04});
    const Vec3d translation{0.12, 0.035, -0.018};

    Mat3x3d rotation1{};
    Mat3x3d rotation2{};
    Mat3x4d projectionMatrix1{};
    Mat3x4d projectionMatrix2{};
    Mat4x4d mappingMatrix{};

    StereoRectify(intrinsics1, distortion1, intrinsics2, distortion2, rotation, translation, 960, 720, rotation1, rotation2, projectionMatrix1, projectionMatrix2, mappingMatrix);

    const cv::Mat cameraMatrix1 = ToCvCameraMatrix(intrinsics1);
    const cv::Mat cameraMatrix2 = ToCvCameraMatrix(intrinsics2);
    const cv::Mat distCoeffs1 = ToCvDistCoeffs(distortion1);
    const cv::Mat distCoeffs2 = ToCvDistCoeffs(distortion2);
    const cv::Mat cvRotation = ToCvRotation(rotation);
    const cv::Mat cvTranslation = ToCvTranslation(translation);

    cv::Mat cvR1;
    cv::Mat cvR2;
    cv::Mat cvP1;
    cv::Mat cvP2;
    cv::Mat cvQ;

    cv::stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, cv::Size(960, 720), cvRotation, cvTranslation, cvR1, cvR2, cvP1, cvP2, cvQ, cv::CALIB_ZERO_DISPARITY, -1);

    const Mat3x3d expectedR1 = ToMat33d(cvR1);
    const Mat3x3d expectedR2 = ToMat33d(cvR2);
    const Mat3x4d expectedP1 = ToMat34d(cvP1);
    const Mat3x4d expectedP2 = ToMat34d(cvP2);
    const Mat4x4d expectedQ = ToMat44d(cvQ);

    ExpectMatrixNear(rotation1, expectedR1, kTolLoose);
    ExpectMatrixNear(rotation2, expectedR2, kTolLoose);
    ExpectMatrix34Near(projectionMatrix1, expectedP1, kTolLoose);
    ExpectMatrix34Near(projectionMatrix2, expectedP2, kTolLoose);
    ExpectMatrix44Near(mappingMatrix, expectedQ, kTolLoose);
}

TEST(GeometryTest, InitUndistortRectifyMapMatchesOpenCV)
{
    const Intrinsics intrinsics{612.5, 598.0, 322.0, 241.5, 0.0};
    const Distortion distortion{-0.013, 0.0045, -6e-4, 3.5e-4, -7.5e-4, 2e-4, -1.1e-4, 6e-5};

    const Mat3x3d rectificationRotation = Exp({-0.02, 0.015, 0.03});
    Mat3x4d projection{};
    projection[0] = {580.0, 0.0, 315.0, 0.0};
    projection[1] = {0.0, 582.0, 238.0, 0.0};
    projection[2] = {0.0, 0.0, 1.0, 0.0};

    constexpr int kWidth = 64;
    constexpr int kHeight = 48;

    Mat mapX;
    Mat mapY;
    ASSERT_TRUE(mapX.Allocate(kHeight, kWidth, 1, sizeof(float), MatLocation::HOST).IsOK());
    ASSERT_TRUE(mapY.Allocate(kHeight, kWidth, 1, sizeof(float), MatLocation::HOST).IsOK());

    InitUndistortRectifyMap(intrinsics, distortion, rectificationRotation, projection, kWidth, kHeight, static_cast<float *>(mapX.Data()), mapX.Stride(), static_cast<float *>(mapY.Data()), mapY.Stride());

    const cv::Mat cameraMatrix = ToCvCameraMatrix(intrinsics);
    const cv::Mat distCoeffs = ToCvDistCoeffs(distortion);
    const cv::Mat cvRectification = ToCvRotation(rectificationRotation);
    cv::Mat cvNewCameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cvNewCameraMatrix.at<double>(0, 0) = projection[0][0];
    cvNewCameraMatrix.at<double>(1, 1) = projection[1][1];
    cvNewCameraMatrix.at<double>(0, 2) = projection[0][2];
    cvNewCameraMatrix.at<double>(1, 2) = projection[1][2];

    cv::Mat cvMapX;
    cv::Mat cvMapY;
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cvRectification, cvNewCameraMatrix, cv::Size(kWidth, kHeight), CV_32FC1, cvMapX, cvMapY);

    const auto *mapXBytes = static_cast<const std::byte *>(mapX.Data());
    const auto *mapYBytes = static_cast<const std::byte *>(mapY.Data());

    double maxErrorX = 0.0;
    double maxErrorY = 0.0;
    for (int v = 0; v < kHeight; ++v)
    {
        const auto *rowX = static_cast<const float *>(static_cast<const void *>(mapXBytes + static_cast<std::size_t>(v) * mapX.Stride()));
        const auto *rowY = static_cast<const float *>(static_cast<const void *>(mapYBytes + static_cast<std::size_t>(v) * mapY.Stride()));
        for (int u = 0; u < kWidth; ++u)
        {
            const double diffX = std::fabs(static_cast<double>(rowX[u]) - cvMapX.at<float>(v, u));
            const double diffY = std::fabs(static_cast<double>(rowY[u]) - cvMapY.at<float>(v, u));
            maxErrorX = std::max(maxErrorX, diffX);
            maxErrorY = std::max(maxErrorY, diffY);
        }
    }

    EXPECT_LT(maxErrorX, kTolLoose);
    EXPECT_LT(maxErrorY, kTolLoose);
}
} // namespace retinify
