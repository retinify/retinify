// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/geometry.hpp"

#include <cmath>

namespace retinify
{
namespace
{
constexpr inline double kEpsilon = 1e-12;
constexpr inline double kPi = 3.141592653589793;

[[nodiscard]] constexpr double Clamp(double value, double lower, double upper) noexcept
{
    return value < lower ? lower : (value > upper ? upper : value);
}

[[nodiscard]] constexpr double Square(double value) noexcept
{
    return value * value;
}

[[nodiscard]] double Reciprocal(double value, double fallback) noexcept
{
    return (std::fabs(value) > kEpsilon) ? (1.0 / value) : fallback;
}
} // namespace

auto Identity() noexcept -> Mat3x3d
{
    return {{{1.0, 0.0, 0.0}, //
             {0.0, 1.0, 0.0}, //
             {0.0, 0.0, 1.0}}};
}

auto Determinant(const Mat3x3d &mat) noexcept -> double
{
    // det(R) = r00(r11 r22 - r12 r21) - r01(r10 r22 - r12 r20) + r02(r10 r21 - r11 r20)
    const auto &row0 = mat[0];
    const auto &row1 = mat[1];
    const auto &row2 = mat[2];
    return row0[0] * (row1[1] * row2[2] - row1[2] * row2[1]) - //
           row0[1] * (row1[0] * row2[2] - row1[2] * row2[0]) + //
           row0[2] * (row1[0] * row2[1] - row1[1] * row2[0]);
}

auto Transpose(const Mat3x3d &mat) noexcept -> Mat3x3d
{
    // (R^T)_{ij} = R_{ji}
    Mat3x3d transposed{};
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            transposed[i][j] = mat[j][i];
        }
    }
    return transposed;
}

auto Multiply(const Mat3x3d &mat, const Vec3d &vec) noexcept -> Vec3d
{
    // y = R x
    Vec3d result{};
    for (int i = 0; i < 3; ++i)
    {
        const auto &row = mat[i];
        result[i] = row[0] * vec[0] + row[1] * vec[1] + row[2] * vec[2];
    }
    return result;
}

auto Multiply(const Mat3x3d &mat1, const Mat3x3d &mat2) noexcept -> Mat3x3d
{
    // C = A B,  c_{ij} = Σ_k a_{ik} b_{kj}
    Mat3x3d result{};
    for (int i = 0; i < 3; ++i)
    {
        const auto &row = mat1[i];
        for (int j = 0; j < 3; ++j)
        {
            result[i][j] = row[0] * mat2[0][j] + row[1] * mat2[1][j] + row[2] * mat2[2][j];
        }
    }
    return result;
}

auto Scale(const Vec3d &vec, double scale) noexcept -> Vec3d
{
    return {vec[0] * scale, vec[1] * scale, vec[2] * scale};
}

auto Length(const Vec3d &vec) noexcept -> double
{
    // ||v|| = sqrt(v·v)
    return std::sqrt(Square(vec[0]) + Square(vec[1]) + Square(vec[2]));
}

auto Normalize(const Vec3d &vec) noexcept -> Vec3d
{
    // v / ||v||  (return zero if degenerate)
    const double n = Length(vec);
    if (n < kEpsilon)
    {
        return {0.0, 0.0, 0.0};
    }
    const double inv = 1.0 / n;
    return {vec[0] * inv, vec[1] * inv, vec[2] * inv};
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
    const double thetaSquared = Square(wx) + Square(wy) + Square(wz);

    Mat3x3d rotation = Identity();
    double coefA = 0.0;
    double coefB = 0.0;

    if (thetaSquared <= kEpsilon)
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

    const double w01 = -wz;
    const double w02 = wy;
    const double w10 = wz;
    const double w12 = -wx;
    const double w20 = -wy;
    const double w21 = wx;

    const double wxx = wx * wx;
    const double wyy = wy * wy;
    const double wzz = wz * wz;
    const double wxy = wx * wy;
    const double wxz = wx * wz;
    const double wyz = wy * wz;

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
    const double trace = rotation[0][0] + rotation[1][1] + rotation[2][2];
    const double cosTheta = Clamp((trace - 1.0) * 0.5, -1.0, 1.0);
    const double theta = std::acos(cosTheta);

    // v = (R − R^T)∨ = 2 sinθ · n  (skew vector)
    const double vx = rotation[2][1] - rotation[1][2];
    const double vy = rotation[0][2] - rotation[2][0];
    const double vz = rotation[1][0] - rotation[0][1];

    // Small angle: ω ≈ 1/2 v (since sinθ ≈ θ)
    if (theta < kEpsilon)
    {
        return {0.5 * vx, 0.5 * vy, 0.5 * vz};
    }

    // Near π: sinθ ~ 0, use diagonal-based axis extraction.
    if (std::fabs(kPi - theta) < 1e-6)
    {
        double ax = std::sqrt(std::max(0.0, (rotation[0][0] + 1.0) * 0.5));
        double ay = std::sqrt(std::max(0.0, (rotation[1][1] + 1.0) * 0.5));
        double az = std::sqrt(std::max(0.0, (rotation[2][2] + 1.0) * 0.5));

        if (ax >= ay && ax >= az)
        {
            const double denom = 4.0 * std::max(ax, kEpsilon);
            ay = (rotation[0][1] + rotation[1][0]) / denom;
            az = (rotation[0][2] + rotation[2][0]) / denom;
        }
        else if (ay >= ax && ay >= az)
        {
            const double denom = 4.0 * std::max(ay, kEpsilon);
            ax = (rotation[0][1] + rotation[1][0]) / denom;
            az = (rotation[1][2] + rotation[2][1]) / denom;
        }
        else
        {
            const double denom = 4.0 * std::max(az, kEpsilon);
            ax = (rotation[0][2] + rotation[2][0]) / denom;
            ay = (rotation[1][2] + rotation[2][1]) / denom;
        }

        // ω = θ n
        const Vec3d axis = Normalize({ax, ay, az});
        return {axis[0] * theta, axis[1] * theta, axis[2] * theta};
    }

    // General case: n = v / ||v||, ω = θ n
    const double vnormSquared = Square(vx) + Square(vy) + Square(vz);
    const double vnorm = std::sqrt(std::max(0.0, vnormSquared));
    if (vnorm < kEpsilon)
    {
        return {0.0, 0.0, 0.0};
    }
    const double scale = theta / vnorm;
    return {scale * vx, scale * vy, scale * vz};
}

auto UndistortPoint(const Intrinsics &intrinsics, const Distortion &distortion, const Point2d &pixel) noexcept -> Point2d
{
    const double invFocalX = Reciprocal(intrinsics.fx, 1.0);
    const double invFocalY = Reciprocal(intrinsics.fy, 1.0);

    const double normX = (pixel[0] - intrinsics.cx) * invFocalX;
    const double normY = (pixel[1] - intrinsics.cy) * invFocalY;

    double undistX = normX;
    double undistY = normY;

    constexpr int kIterationCount = 5;
    for (int iter = 0; iter < kIterationCount; ++iter)
    {
        // Brown-Conrady rational model inversion with r^2 = x^2 + y^2:
        // radialScale = (1 + k4 r^2 + k5 r^4 + k6 r^6) / (1 + k1 r^2 + k2 r^4 + k3 r^6)
        // Delta tangential = (2 p1 xy + p2 (r^2 + 2 x^2),
        //                     p1 (r^2 + 2 y^2) + 2 p2 xy)
        const double rSquared = Square(undistX) + Square(undistY);
        const double rFourth = rSquared * rSquared;
        const double rSixth = rFourth * rSquared;

        const double radialNumerator = 1.0 + distortion.k4 * rSquared + distortion.k5 * rFourth + distortion.k6 * rSixth;
        const double radialDenominator = 1.0 + distortion.k1 * rSquared + distortion.k2 * rFourth + distortion.k3 * rSixth;
        const double radialScale = (std::fabs(radialDenominator) > kEpsilon) ? (radialNumerator / radialDenominator) : 1.0;
        if (radialScale < 0.0)
        {
            undistX = normX;
            undistY = normY;
            break;
        }

        const double twoXY = 2.0 * undistX * undistY;
        const double xSquared = Square(undistX);
        const double ySquared = Square(undistY);
        const double deltaX = distortion.p1 * twoXY + distortion.p2 * (rSquared + 2.0 * xSquared);
        const double deltaY = distortion.p1 * (rSquared + 2.0 * ySquared) + distortion.p2 * twoXY;

        undistX = (normX - deltaX) * radialScale;
        undistY = (normY - deltaY) * radialScale;
    }
    return {undistX, undistY};
}

namespace
{
static auto ComputeRectifyingRotation(const Mat3x3d &rotation) noexcept -> Mat3x3d
{
    // Split relative rotation equally: R_rect = exp(-0.5 * log(R))
    const Vec3d omega = Log(rotation);
    return Exp(Scale(omega, -0.5));
}

static auto DetermineDominantAxis(const Vec3d &translation) noexcept -> int
{
    // argmax_i |T_i|
    return (std::fabs(translation[0]) > std::fabs(translation[1])) ? 0 : 1;
}

static auto BuildAxisVector(int idx, double direction) noexcept -> Vec3d
{
    Vec3d axis{0.0, 0.0, 0.0};
    axis[idx] = direction;
    return axis;
}

static auto ComputeBaselineAlignment(const Vec3d &translation, int axisIdx) noexcept -> Mat3x3d
{
    // Rotate translation onto +/- e_axis using axis = T x target, angle = acos(|T_axis| / ||T||)
    const double component = translation[axisIdx];
    const double length = Length(translation);
    const Vec3d target = BuildAxisVector(axisIdx, component >= 0.0 ? 1.0 : -1.0);
    const Vec3d cross = Cross(translation, target);
    const double crossLength = Length(cross);
    if (crossLength <= kEpsilon || length <= kEpsilon)
    {
        return Identity();
    }
    const double arg = Clamp(std::fabs(component) / length, -1.0, 1.0);
    const double angle = std::acos(arg);
    const double scale = angle / crossLength;
    return Exp(Scale(cross, scale));
}

static auto ComputePrincipalPoint(const Intrinsics &intrinsics, const Distortion &distortion, const Mat3x3d &rectifiedRotation, //
                                  double newFocalLength, double width, double height) noexcept -> Point2d
{
    // Principal shift: c = (w-1, h-1)/2 - f * average( rectified_xy / rectified_z ) over undistorted corners
    const std::array<Point2d, 4> corners{Point2d{0.0, 0.0}, Point2d{width - 1.0, 0.0}, Point2d{0.0, height - 1.0}, Point2d{width - 1.0, height - 1.0}};

    double accumulatedX = 0.0;
    double accumulatedY = 0.0;
    for (const auto &corner : corners)
    {
        const Point2d undistorted2D = UndistortPoint(intrinsics, distortion, corner);
        const Vec3d undistorted3D{undistorted2D[0], undistorted2D[1], 1.0};
        const Vec3d rectifiedPoint = Multiply(rectifiedRotation, undistorted3D);
        const double inverseDepth = Reciprocal(rectifiedPoint[2], 1.0);
        accumulatedX += newFocalLength * rectifiedPoint[0] * inverseDepth;
        accumulatedY += newFocalLength * rectifiedPoint[1] * inverseDepth;
    }

    const double halfWidth = (width - 1.0) * 0.5;
    const double halfHeight = (height - 1.0) * 0.5;
    return {halfWidth - accumulatedX * 0.25, halfHeight - accumulatedY * 0.25};
}

static auto BuildProjectionMatrix(double focal, const Point2d &principal) noexcept -> Mat3x4d
{
    // P = [[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0]]
    Mat3x4d projection{};
    projection[0][0] = focal;
    projection[0][2] = principal[0];
    projection[1][1] = focal;
    projection[1][2] = principal[1];
    projection[2][2] = 1.0;
    return projection;
}
} // namespace

auto StereoRectify(const Intrinsics &intrinsics1, const Distortion &distortion1, //
                   const Intrinsics &intrinsics2, const Distortion &distortion2, //
                   const Mat3x3d &rotation, const Vec3d &translation,            //
                   std::uint32_t imageWidth, std::uint32_t imageHeight,          //
                   Mat3x3d &rotation1, Mat3x3d &rotation2,                       //
                   Mat3x4d &projectionMatrix1, Mat3x4d &projectionMatrix2,       //
                   Mat4x4d &mappingMatrix) noexcept -> void
{
    // Stereo rectification: rotation1 = R_align R_rect^T, rotation2 = R_align R_rect,
    // share focal f = 0.5 * ((fx1+fx2) if axis=y else (fy1+fy2)), and set disparity scale via baseline
    const Mat3x3d rectifyingRotation = ComputeRectifyingRotation(rotation);
    const Vec3d rotatedTranslation = Multiply(rectifyingRotation, translation);
    const int axisIdx = DetermineDominantAxis(rotatedTranslation);
    const Mat3x3d baselineAlignment = ComputeBaselineAlignment(rotatedTranslation, axisIdx);

    const Mat3x3d rotationTranspose = Transpose(rectifyingRotation);
    rotation1 = Multiply(baselineAlignment, rotationTranspose);
    rotation2 = Multiply(baselineAlignment, rectifyingRotation);

    const Vec3d rectifiedTranslation = Multiply(rotation2, translation);
    const double newFocalScale = 0.5;
    const double newFocalLength = (axisIdx == 0) ? (intrinsics1.fy + intrinsics2.fy) * newFocalScale : (intrinsics1.fx + intrinsics2.fx) * newFocalScale;

    const Point2d principal1 = ComputePrincipalPoint(intrinsics1, distortion1, rotation1, newFocalLength, static_cast<double>(imageWidth), static_cast<double>(imageHeight));
    const Point2d principal2 = ComputePrincipalPoint(intrinsics2, distortion2, rotation2, newFocalLength, static_cast<double>(imageWidth), static_cast<double>(imageHeight));
    const double cxAvg = 0.5 * (principal1[0] + principal2[0]);
    const double cyAvg = 0.5 * (principal1[1] + principal2[1]);
    const Point2d principalAvg{cxAvg, cyAvg};

    projectionMatrix1 = BuildProjectionMatrix(newFocalLength, principalAvg);
    projectionMatrix2 = BuildProjectionMatrix(newFocalLength, principalAvg);

    const double baselineComponent = (axisIdx == 0) ? rectifiedTranslation[0] : rectifiedTranslation[1];
    const double translationOffset = baselineComponent * newFocalLength;
    if (axisIdx == 0)
    {
        projectionMatrix2[0][3] = translationOffset;
    }
    else
    {
        projectionMatrix2[1][3] = translationOffset;
    }

    mappingMatrix = Mat4x4d{};
    mappingMatrix[0][0] = 1.0;
    mappingMatrix[1][1] = 1.0;
    mappingMatrix[0][3] = -principalAvg[0];
    mappingMatrix[1][3] = -principalAvg[1];
    mappingMatrix[2][3] = newFocalLength;
    mappingMatrix[3][2] = (std::fabs(baselineComponent) > kEpsilon) ? (-1.0 / baselineComponent) : 0.0;
    mappingMatrix[3][3] = 0.0;
}

auto InitUndistortRectifyMap(const Intrinsics &intrinsics, const Distortion &distortion, //
                             const Mat3x3d &rotation, const Mat3x4d &projectionMatrix,   //
                             std::uint32_t imageWidth, std::uint32_t imageHeight,        //
                             float *mapx, std::size_t mapxStride,                        //
                             float *mapy, std::size_t mapyStride) noexcept -> void
{
    if (mapx == nullptr || mapy == nullptr)
    {
        return;
    }

    const Mat3x3d rotationInverse = Transpose(rotation);
    const auto &projRow0 = projectionMatrix[0];
    const auto &projRow1 = projectionMatrix[1];
    const double invRectifiedFocalX = Reciprocal(projRow0[0], 0.0);
    const double invRectifiedFocalY = Reciprocal(projRow1[1], 0.0);
    const double rectifiedPrincipalX = projRow0[2];
    const double rectifiedPrincipalY = projRow1[2];

    auto *mapxBytes = static_cast<unsigned char *>(static_cast<void *>(mapx));
    auto *mapyBytes = static_cast<unsigned char *>(static_cast<void *>(mapy));

    for (std::uint32_t v = 0; v < imageHeight; ++v)
    {
        const std::size_t offsetX = static_cast<std::size_t>(v) * mapxStride;
        const std::size_t offsetY = static_cast<std::size_t>(v) * mapyStride;
        auto *mapxRow = static_cast<float *>(static_cast<void *>(mapxBytes + offsetX));
        auto *mapyRow = static_cast<float *>(static_cast<void *>(mapyBytes + offsetY));
        const double rectifiedY = (static_cast<double>(v) - rectifiedPrincipalY) * invRectifiedFocalY;
        for (std::uint32_t u = 0; u < imageWidth; ++u)
        {
            const double rectifiedX = (static_cast<double>(u) - rectifiedPrincipalX) * invRectifiedFocalX;
            const Vec3d rectifiedPoint{rectifiedX, rectifiedY, 1.0};
            const Vec3d cameraPoint = Multiply(rotationInverse, rectifiedPoint);

            const double inverseDepth = Reciprocal(cameraPoint[2], 0.0);
            const double normalizedX = cameraPoint[0] * inverseDepth;
            const double normalizedY = cameraPoint[1] * inverseDepth;

            const double radiusSquared = Square(normalizedX) + Square(normalizedY);
            const double radiusFourth = radiusSquared * radiusSquared;
            const double radiusSixth = radiusFourth * radiusSquared;
            const double radialNumerator = 1.0 + distortion.k1 * radiusSquared + distortion.k2 * radiusFourth + distortion.k3 * radiusSixth;
            const double radialDenominator = 1.0 + distortion.k4 * radiusSquared + distortion.k5 * radiusFourth + distortion.k6 * radiusSixth;
            const double radialScale = (std::fabs(radialDenominator) > kEpsilon) ? (radialNumerator / radialDenominator) : 1.0;

            const double twoNormalizedXY = 2.0 * normalizedX * normalizedY;
            const double normalizedXSquared = Square(normalizedX);
            const double normalizedYSquared = Square(normalizedY);

            const double distortedNormalizedX = normalizedX * radialScale + distortion.p1 * twoNormalizedXY + distortion.p2 * (radiusSquared + 2.0 * normalizedXSquared);
            const double distortedNormalizedY = normalizedY * radialScale + distortion.p1 * (radiusSquared + 2.0 * normalizedYSquared) + distortion.p2 * twoNormalizedXY;

            const double uDistorted = intrinsics.fx * distortedNormalizedX + intrinsics.skew * distortedNormalizedY + intrinsics.cx;
            const double vDistorted = intrinsics.fy * distortedNormalizedY + intrinsics.cy;
            mapxRow[u] = static_cast<float>(uDistorted);
            mapyRow[u] = static_cast<float>(vDistorted);
        }
    }
}
} // namespace retinify
