// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/geometry.hpp"
#include "retinify/logging.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace retinify
{
namespace
{
constexpr inline double kEpsilon = 1e-12;
constexpr inline double kPi = 3.141592653589793;
constexpr inline std::size_t kMat33RowCount = 3;
constexpr inline std::size_t kMat33ColCount = 3;
} // namespace

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
    Mat3x3d transposed{};
    for (std::size_t row = 0; row < kMat33RowCount; ++row)
    {
        for (std::size_t col = 0; col < kMat33ColCount; ++col)
        {
            transposed[row][col] = mat[col][row];
        }
    }
    return transposed;
}

auto Add(const Mat3x3d &mat1, const Mat3x3d &mat2) noexcept -> Mat3x3d
{
    Mat3x3d result{};
    for (std::size_t row = 0; row < kMat33RowCount; ++row)
    {
        for (std::size_t col = 0; col < kMat33ColCount; ++col)
        {
            result[row][col] = mat1[row][col] + mat2[row][col];
        }
    }
    return result;
}

auto Multiply(const Mat3x3d &mat, const Vec3d &vec) noexcept -> Vec3d
{
    Vec3d result{};
    for (std::size_t row = 0; row < kMat33RowCount; ++row)
    {
        const auto &matRow = mat[row];
        result[row] = matRow[0] * vec[0] + matRow[1] * vec[1] + matRow[2] * vec[2];
    }
    return result;
}

auto Multiply(const Mat3x3d &mat1, const Mat3x3d &mat2) noexcept -> Mat3x3d
{
    Mat3x3d result{};
    for (std::size_t row = 0; row < kMat33RowCount; ++row)
    {
        const auto &rowValues = mat1[row];
        for (std::size_t col = 0; col < kMat33ColCount; ++col)
        {
            result[row][col] = rowValues[0] * mat2[0][col] + rowValues[1] * mat2[1][col] + rowValues[2] * mat2[2][col];
        }
    }
    return result;
}

auto Multiply(const Mat3x3d &mat, double scale) noexcept -> Mat3x3d
{
    Mat3x3d result{};
    for (std::size_t row = 0; row < kMat33RowCount; ++row)
    {
        for (std::size_t col = 0; col < kMat33ColCount; ++col)
        {
            result[row][col] = mat[row][col] * scale;
        }
    }
    return result;
}

auto Multiply(const Vec3d &vec, double scale) noexcept -> Vec3d
{
    return {vec[0] * scale, vec[1] * scale, vec[2] * scale};
}

auto Length(const Vec3d &vec) noexcept -> double
{
    return std::sqrt(Dot(vec, vec));
}

auto Normalize(const Vec3d &vec) noexcept -> Vec3d
{
    const double n = Length(vec);
    if (n < kEpsilon)
    {
        return {0.0, 0.0, 0.0};
    }
    const double inv = 1.0 / n;
    return {vec[0] * inv, vec[1] * inv, vec[2] * inv};
}

auto Dot(const Vec3d &vec1, const Vec3d &vec2) noexcept -> double
{
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
}

auto Cross(const Vec3d &vec1, const Vec3d &vec2) noexcept -> Vec3d
{
    return {vec1[1] * vec2[2] - vec1[2] * vec2[1], //
            vec1[2] * vec2[0] - vec1[0] * vec2[2], //
            vec1[0] * vec2[1] - vec1[1] * vec2[0]};
}

auto Hat(const Vec3d &vec) noexcept -> Mat3x3d
{
    return {{{0.0, -vec[2], vec[1]}, //
             {vec[2], 0.0, -vec[0]}, //
             {-vec[1], vec[0], 0.0}}};
}

auto Vee(const Mat3x3d &mat) noexcept -> Vec3d
{
    return {0.5 * (mat[2][1] - mat[1][2]), //
            0.5 * (mat[0][2] - mat[2][0]), //
            0.5 * (mat[1][0] - mat[0][1])};
}

auto Exp(const Vec3d &vec) noexcept -> Mat3x3d
{
    const double thetaSquared = Dot(vec, vec);
    double coefA = 0.0;
    double coefB = 0.0;

    if (thetaSquared <= kEpsilon)
    {
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
    const Mat3x3d skew = Hat(vec);
    const Mat3x3d skewSquared = Multiply(skew, skew);

    Mat3x3d rotation = Identity();
    rotation = Add(rotation, Multiply(skew, coefA));
    rotation = Add(rotation, Multiply(skewSquared, coefB));
    return rotation;
}

auto Log(const Mat3x3d &mat) noexcept -> Vec3d
{
    const double trace = mat[0][0] + mat[1][1] + mat[2][2];
    const double cosTheta = std::clamp((trace - 1.0) * 0.5, -1.0, 1.0);
    const double theta = std::acos(cosTheta);

    const Vec3d skewVector{mat[2][1] - mat[1][2], mat[0][2] - mat[2][0], mat[1][0] - mat[0][1]};

    if (theta < kEpsilon)
    {
        return Multiply(skewVector, 0.5);
    }

    if (std::fabs(kPi - theta) < kEpsilon)
    {
        double ax = std::sqrt(std::max(0.0, (mat[0][0] + 1.0) * 0.5));
        double ay = std::sqrt(std::max(0.0, (mat[1][1] + 1.0) * 0.5));
        double az = std::sqrt(std::max(0.0, (mat[2][2] + 1.0) * 0.5));

        if (ax >= ay && ax >= az)
        {
            const double denom = 4.0 * std::max(ax, kEpsilon);
            ay = (mat[0][1] + mat[1][0]) / denom;
            az = (mat[0][2] + mat[2][0]) / denom;
        }
        else if (ay >= ax && ay >= az)
        {
            const double denom = 4.0 * std::max(ay, kEpsilon);
            ax = (mat[0][1] + mat[1][0]) / denom;
            az = (mat[1][2] + mat[2][1]) / denom;
        }
        else
        {
            const double denom = 4.0 * std::max(az, kEpsilon);
            ax = (mat[0][2] + mat[2][0]) / denom;
            ay = (mat[1][2] + mat[2][1]) / denom;
        }

        Vec3d axisCandidate{ax, ay, az};
        Vec3d axis = Normalize(axisCandidate);
        return Multiply(axis, theta);
    }

    const double skewVectorNormSquared = Dot(skewVector, skewVector);
    const double skewVectorNorm = std::sqrt(std::max(0.0, skewVectorNormSquared));
    if (skewVectorNorm < kEpsilon)
    {
        return {0.0, 0.0, 0.0};
    }
    const double scale = theta / skewVectorNorm;
    return Multiply(skewVector, scale);
}

namespace
{
[[nodiscard]] auto Square(double value) noexcept -> double
{
    return value * value;
}

[[nodiscard]] auto Reciprocal(double value, double fallback) noexcept -> double
{
    return (std::fabs(value) > kEpsilon) ? (1.0 / value) : fallback;
}

[[nodiscard]] auto ComputeRadialDistortionScale(double radiusSquared, double coeff2, double coeff4, double coeff6) noexcept -> double
{
    const double radiusFourth = radiusSquared * radiusSquared;
    const double radiusSixth = radiusFourth * radiusSquared;
    return 1.0 + coeff2 * radiusSquared + coeff4 * radiusFourth + coeff6 * radiusSixth;
}

[[nodiscard]] auto ComputeInverseSqrtRotation(const Mat3x3d &rotation) noexcept -> Mat3x3d
{
    const Vec3d omega = Log(rotation);
    return Exp(Multiply(omega, -0.5));
}

enum class BaselineAxis : std::uint8_t
{
    X = 0,
    Y = 1
};

[[nodiscard]] auto DetermineBaselineAxis(const Vec3d &translation) noexcept -> BaselineAxis
{
    return (std::fabs(translation[0]) > std::fabs(translation[1])) ? BaselineAxis::X : BaselineAxis::Y;
}

[[nodiscard]] auto ToBaselineAxisIndex(BaselineAxis axis) noexcept -> std::uint8_t
{
    return static_cast<std::uint8_t>(axis);
}

[[nodiscard]] auto ComputeBaselineAxisVector(BaselineAxis axis, double direction) noexcept -> Vec3d
{
    Vec3d axisVector{0.0, 0.0, 0.0};
    axisVector[ToBaselineAxisIndex(axis)] = direction;
    return axisVector;
}

[[nodiscard]] auto ComputeBaselineAxisAlignmentRotation(const Vec3d &translation, BaselineAxis axis) noexcept -> Mat3x3d
{
    const int axisIndex = ToBaselineAxisIndex(axis);
    const double component = translation[axisIndex];
    const double length = Length(translation);
    const Vec3d targetAxisVector = ComputeBaselineAxisVector(axis, component >= 0.0 ? 1.0 : -1.0);
    const Vec3d cross = Cross(translation, targetAxisVector);
    const double crossLength = Length(cross);
    if (crossLength <= kEpsilon || length <= kEpsilon)
    {
        return Identity();
    }
    const double arg = std::clamp(std::fabs(component) / length, -1.0, 1.0);
    const double angle = std::acos(arg);
    const double scale = angle / crossLength;
    return Exp(Multiply(cross, scale));
}

[[nodiscard]] auto ComputePrincipalPoint(const PinholeIntrinsics &intrinsics, const DistortionCoefficients &distortion, const Mat3x3d &rectifiedRotation, double newFocalLength, double width, double height) noexcept -> Point2d
{
    const std::array<Point2d, 4> imageCorners{Point2d{0.0, 0.0}, Point2d{width - 1.0, 0.0}, Point2d{0.0, height - 1.0}, Point2d{width - 1.0, height - 1.0}};

    double accumulatedX = 0.0;
    double accumulatedY = 0.0;
    for (const auto &imageCorner : imageCorners)
    {
        const Point2d undistorted2D = UndistortPoint(intrinsics, distortion, imageCorner);
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

[[nodiscard]] auto ComputeCameraMatrix(double focalLength, const Point2d &principalPoint) noexcept -> Mat3x3d
{
    Mat3x3d camera{};
    camera[0][0] = focalLength;
    camera[0][1] = 0.0;
    camera[0][2] = principalPoint[0];
    camera[1][0] = 0.0;
    camera[1][1] = focalLength;
    camera[1][2] = principalPoint[1];
    camera[2][0] = 0.0;
    camera[2][1] = 0.0;
    camera[2][2] = 1.0;
    return camera;
}

[[nodiscard]] auto ComputeProjectionMatrix(double focalLength, const Point2d &principalPoint) noexcept -> Mat3x4d
{
    Mat3x4d projection{};
    projection[0][0] = focalLength;
    projection[0][2] = principalPoint[0];
    projection[1][1] = focalLength;
    projection[1][2] = principalPoint[1];
    projection[2][2] = 1.0;
    return projection;
}

constexpr double kInfinity = std::numeric_limits<double>::infinity();

[[nodiscard]] auto ComputeRobustRatio(double numerator, double denominator) noexcept -> double
{
    if (std::fabs(denominator) <= kEpsilon)
    {
        if (std::fabs(numerator) <= kEpsilon)
        {
            return 0.0;
        }
        const double sign = (numerator >= 0.0) ? 1.0 : -1.0;
        return sign * kInfinity;
    }
    return numerator / denominator;
}

[[nodiscard]] constexpr auto IsBorderIndex(int index, int lastIndex) noexcept -> bool
{
    return index == 0 || index == lastIndex;
}

auto ComputeRectifiedInnerOuterRectangles(const PinholeIntrinsics &intrinsics, const DistortionCoefficients &distortion, const Mat3x3d &rectifiedRotation, const Mat3x3d &newCameraMatrix, std::uint32_t imageWidth, std::uint32_t imageHeight, Rect2d &inner, Rect2d &outer) noexcept -> void
{
    constexpr int kGridSize = 9;
    constexpr int kLastIndex = kGridSize - 1;
    const double width = static_cast<double>(imageWidth);
    const double height = static_cast<double>(imageHeight);
    const double maxX = width - 1.0;
    const double maxY = height - 1.0;
    const double stepX = (kGridSize > 1) ? (maxX / static_cast<double>(kLastIndex)) : 0.0;
    const double stepY = (kGridSize > 1) ? (maxY / static_cast<double>(kLastIndex)) : 0.0;

    double innerMinX = -kInfinity;
    double innerMaxX = kInfinity;
    double innerMinY = -kInfinity;
    double innerMaxY = kInfinity;
    double outerMinX = kInfinity;
    double outerMaxX = -kInfinity;
    double outerMinY = kInfinity;
    double outerMaxY = -kInfinity;

    const auto projectRectifiedNormalizedPoint = [&](double rectifiedNormalizedX, double rectifiedNormalizedY) noexcept -> Point2d {
        const double numeratorX = newCameraMatrix[0][0] * rectifiedNormalizedX + newCameraMatrix[0][1] * rectifiedNormalizedY + newCameraMatrix[0][2];
        const double numeratorY = newCameraMatrix[1][0] * rectifiedNormalizedX + newCameraMatrix[1][1] * rectifiedNormalizedY + newCameraMatrix[1][2];
        const double denominator = newCameraMatrix[2][0] * rectifiedNormalizedX + newCameraMatrix[2][1] * rectifiedNormalizedY + newCameraMatrix[2][2];
        const double scale = Reciprocal(denominator, 1.0);
        return {numeratorX * scale, numeratorY * scale};
    };

    const auto updateBoundaryExtents = [&](const Point2d &mappedPoint, bool onLeftEdge, bool onRightEdge, bool onTopEdge, bool onBottomEdge) noexcept {
        const double mappedX = mappedPoint[0];
        const double mappedY = mappedPoint[1];

        outerMinX = std::min(outerMinX, mappedX);
        outerMaxX = std::max(outerMaxX, mappedX);
        outerMinY = std::min(outerMinY, mappedY);
        outerMaxY = std::max(outerMaxY, mappedY);

        if (onLeftEdge)
        {
            innerMinX = std::max(innerMinX, mappedX);
        }

        if (onRightEdge)
        {
            innerMaxX = std::min(innerMaxX, mappedX);
        }

        if (onTopEdge)
        {
            innerMinY = std::max(innerMinY, mappedY);
        }

        if (onBottomEdge)
        {
            innerMaxY = std::min(innerMaxY, mappedY);
        }
    };

    for (int gridY = 0; gridY < kGridSize; ++gridY)
    {
        const bool onYBorder = IsBorderIndex(gridY, kLastIndex);
        const bool onTopEdge = (gridY == 0);
        const bool onBottomEdge = (gridY == kLastIndex);
        const double pixelY = stepY * static_cast<double>(gridY);

        for (int gridX = 0; gridX < kGridSize; ++gridX)
        {
            const bool onXBorder = IsBorderIndex(gridX, kLastIndex);
            if (!onXBorder && !onYBorder)
            {
                continue;
            }

            const double pixelX = stepX * static_cast<double>(gridX);
            const Point2d undistortedPoint = UndistortPoint(intrinsics, distortion, {pixelX, pixelY});
            const Vec3d undistortedPoint3D{undistortedPoint[0], undistortedPoint[1], 1.0};
            const Vec3d rectifiedUndistortedPoint3D = Multiply(rectifiedRotation, undistortedPoint3D);
            const double inverseDepth = Reciprocal(rectifiedUndistortedPoint3D[2], 0.0);
            const double rectifiedNormalizedX = rectifiedUndistortedPoint3D[0] * inverseDepth;
            const double rectifiedNormalizedY = rectifiedUndistortedPoint3D[1] * inverseDepth;

            const Point2d mappedPoint = projectRectifiedNormalizedPoint(rectifiedNormalizedX, rectifiedNormalizedY);
            const bool onLeftEdge = (gridX == 0);
            const bool onRightEdge = (gridX == kLastIndex);
            updateBoundaryExtents(mappedPoint, onLeftEdge, onRightEdge, onTopEdge, onBottomEdge);
        }
    }

    if (!std::isfinite(innerMinX) || !std::isfinite(innerMaxX) || !std::isfinite(innerMinY) || !std::isfinite(innerMaxY))
    {
        inner = {};
    }
    else
    {
        inner = {innerMinX, innerMinY, std::max(0.0, innerMaxX - innerMinX), std::max(0.0, innerMaxY - innerMinY)};
    }

    if (!std::isfinite(outerMinX) || !std::isfinite(outerMaxX) || !std::isfinite(outerMinY) || !std::isfinite(outerMaxY))
    {
        outer = {};
    }
    else
    {
        outer = {outerMinX, outerMinY, std::max(0.0, outerMaxX - outerMinX), std::max(0.0, outerMaxY - outerMinY)};
    }
}

[[nodiscard]] auto ComputeRectifiedFocalLengthScale(const PinholeIntrinsics &intrinsics1, const DistortionCoefficients &distortion1, const Mat3x3d &rectifiedRotation1, const PinholeIntrinsics &intrinsics2, const DistortionCoefficients &distortion2, const Mat3x3d &rectifiedRotation2, double focalLength, const Point2d &principalPoint1, const Point2d &principalPoint2, std::uint32_t imageWidth, std::uint32_t imageHeight, double alpha) noexcept -> double
{
    if (alpha < 0.0)
    {
        return 1.0;
    }

    const double clampedAlpha = std::clamp(alpha, 0.0, 1.0);

    const Mat3x3d cameraMatrix1 = ComputeCameraMatrix(focalLength, principalPoint1);
    const Mat3x3d cameraMatrix2 = ComputeCameraMatrix(focalLength, principalPoint2);

    Rect2d inner1{};
    Rect2d outer1{};
    Rect2d inner2{};
    Rect2d outer2{};
    ComputeRectifiedInnerOuterRectangles(intrinsics1, distortion1, rectifiedRotation1, cameraMatrix1, imageWidth, imageHeight, inner1, outer1);
    ComputeRectifiedInnerOuterRectangles(intrinsics2, distortion2, rectifiedRotation2, cameraMatrix2, imageWidth, imageHeight, inner2, outer2);

    const double width = static_cast<double>(imageWidth);
    const double height = static_cast<double>(imageHeight);

    const auto computeInnerScale = [&](const Rect2d &inner, const Point2d &principal) noexcept -> double {
        const double cx = principal[0];
        const double cy = principal[1];
        const double scaleLeft = ComputeRobustRatio(cx, cx - inner.x);
        const double scaleRight = ComputeRobustRatio(width - 1.0 - cx, inner.x + inner.width - cx);
        const double scaleTop = ComputeRobustRatio(cy, cy - inner.y);
        const double scaleBottom = ComputeRobustRatio(height - 1.0 - cy, inner.y + inner.height - cy);
        return std::max(std::max(scaleLeft, scaleRight), std::max(scaleTop, scaleBottom));
    };

    const auto computeOuterScale = [&](const Rect2d &outer, const Point2d &principal) noexcept -> double {
        const double cx = principal[0];
        const double cy = principal[1];
        const double scaleLeft = ComputeRobustRatio(cx, cx - outer.x);
        const double scaleRight = ComputeRobustRatio(width - 1.0 - cx, outer.x + outer.width - cx);
        const double scaleTop = ComputeRobustRatio(cy, cy - outer.y);
        const double scaleBottom = ComputeRobustRatio(height - 1.0 - cy, outer.y + outer.height - cy);
        return std::min(std::min(scaleLeft, scaleRight), std::min(scaleTop, scaleBottom));
    };

    const double innerScale1 = computeInnerScale(inner1, principalPoint1);
    const double innerScale2 = computeInnerScale(inner2, principalPoint2);
    const double s0 = std::max(innerScale1, innerScale2);

    const double outerScale1 = computeOuterScale(outer1, principalPoint1);
    const double outerScale2 = computeOuterScale(outer2, principalPoint2);
    const double s1 = std::min(outerScale1, outerScale2);

    const double scale = s0 * (1.0 - clampedAlpha) + s1 * clampedAlpha;
    if (!std::isfinite(scale) || scale <= 0.0)
    {
        return 1.0;
    }
    return scale;
}
} // namespace

auto UndistortPoint(const PinholeIntrinsics &intrinsics, const DistortionCoefficients &distortion, const Point2d &point) noexcept -> Point2d
{
    const double inverseFocalX = Reciprocal(intrinsics.fx, 1.0);
    const double inverseFocalY = Reciprocal(intrinsics.fy, 1.0);

    const double normalizedX = (point[0] - intrinsics.cx) * inverseFocalX;
    const double normalizedY = (point[1] - intrinsics.cy) * inverseFocalY;

    double undistortedX = normalizedX;
    double undistortedY = normalizedY;

    constexpr int kIterationCount = 5;
    for (int iter = 0; iter < kIterationCount; ++iter)
    {
        const double radiusSquared = Square(undistortedX) + Square(undistortedY);
        const double radialNumerator = ComputeRadialDistortionScale(radiusSquared, distortion.k4, distortion.k5, distortion.k6);
        const double radialDenominator = ComputeRadialDistortionScale(radiusSquared, distortion.k1, distortion.k2, distortion.k3);
        const double inverseRadialDenominator = Reciprocal(radialDenominator, 0.0);
        const double radialScale = (inverseRadialDenominator != 0.0) ? radialNumerator * inverseRadialDenominator : 1.0;

        const double twiceUndistortedXY = 2.0 * undistortedX * undistortedY;
        const double undistortedXSquared = Square(undistortedX);
        const double undistortedYSquared = Square(undistortedY);
        const double deltaX = distortion.p1 * twiceUndistortedXY + distortion.p2 * (radiusSquared + 2.0 * undistortedXSquared);
        const double deltaY = distortion.p1 * (radiusSquared + 2.0 * undistortedYSquared) + distortion.p2 * twiceUndistortedXY;

        undistortedX = (normalizedX - deltaX) * radialScale;
        undistortedY = (normalizedY - deltaY) * radialScale;
    }
    return {undistortedX, undistortedY};
}

auto DistortPoint(const PinholeIntrinsics &intrinsics, const DistortionCoefficients &distortion, const Point2d &point) noexcept -> Point2d
{
    const double undistortedX = point[0];
    const double undistortedY = point[1];

    const double radiusSquared = Square(undistortedX) + Square(undistortedY);
    const double radialNumerator = ComputeRadialDistortionScale(radiusSquared, distortion.k1, distortion.k2, distortion.k3);
    const double radialDenominator = ComputeRadialDistortionScale(radiusSquared, distortion.k4, distortion.k5, distortion.k6);
    const double inverseRadialDenominator = Reciprocal(radialDenominator, 0.0);
    const double radial = (inverseRadialDenominator != 0.0) ? radialNumerator * inverseRadialDenominator : 1.0;

    const double twiceUndistortedXY = 2.0 * undistortedX * undistortedY;
    const double undistortedXSquared = Square(undistortedX);
    const double undistortedYSquared = Square(undistortedY);
    const double deltaX = distortion.p1 * twiceUndistortedXY + distortion.p2 * (radiusSquared + 2.0 * undistortedXSquared);
    const double deltaY = distortion.p1 * (radiusSquared + 2.0 * undistortedYSquared) + distortion.p2 * twiceUndistortedXY;

    const double distortedX = undistortedX * radial + deltaX;
    const double distortedY = undistortedY * radial + deltaY;

    return {distortedX * intrinsics.fx + intrinsics.cx, distortedY * intrinsics.fy + intrinsics.cy};
}

auto Undistort(const PinholeIntrinsics &intrinsics, const DistortionCoefficients &distortion, std::uint32_t imageWidth, std::uint32_t imageHeight, const std::uint8_t *src, std::size_t srcStride, std::uint8_t *dst, std::size_t dstStride) noexcept -> Status
{
    if (src == nullptr || dst == nullptr)
    {
        LogError("source and destination pointers must not be null.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    if ((imageWidth == 0U) || (imageHeight == 0U))
    {
        LogError("imageWidth and imageHeight must be greater than zero.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    const std::size_t requiredStride = static_cast<std::size_t>(imageWidth) * sizeof(std::uint8_t);
    if (srcStride < requiredStride)
    {
        LogError("src stride is too small for the given image width.");
        LogStrideError(srcStride, requiredStride);
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    if (dstStride < requiredStride)
    {
        LogError("dst stride is too small for the given image width.");
        LogStrideError(dstStride, requiredStride);
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    constexpr int kInterpolationBits = 5;
    constexpr int kInterpolationTabSize = 1 << kInterpolationBits;
    constexpr int kInterpolationTabMask = kInterpolationTabSize - 1;
    constexpr int kInterpolationTabSize2 = kInterpolationTabSize * kInterpolationTabSize;

    const double inverseFocalX = Reciprocal(intrinsics.fx, 1.0);
    const double inverseFocalY = Reciprocal(intrinsics.fy, 1.0);
    const double skew = intrinsics.skew;
    const int width = static_cast<int>(imageWidth);
    const int height = static_cast<int>(imageHeight);
    const auto *srcBytes = reinterpret_cast<const unsigned char *>(src);
    auto *dstBytes = reinterpret_cast<unsigned char *>(dst);

    const auto samplePixel = [&](int x, int y) noexcept -> int {
        if (x < 0 || y < 0 || x >= width || y >= height)
        {
            return 0;
        }
        const std::size_t offset = static_cast<std::size_t>(y) * srcStride + static_cast<std::size_t>(x);
        return static_cast<int>(srcBytes[offset]);
    };

    for (std::uint32_t v = 0; v < imageHeight; ++v)
    {
        const std::size_t offsetDst = static_cast<std::size_t>(v) * dstStride;
        auto *dstRow = reinterpret_cast<std::uint8_t *>(dstBytes + offsetDst);
        const double undistortedY = (static_cast<double>(v) - intrinsics.cy) * inverseFocalY;
        for (std::uint32_t u = 0; u < imageWidth; ++u)
        {
            const double undistortedX = (static_cast<double>(u) - intrinsics.cx - skew * undistortedY) * inverseFocalX;

            const double radiusSquared = Square(undistortedX) + Square(undistortedY);
            const double radialNumerator = ComputeRadialDistortionScale(radiusSquared, distortion.k1, distortion.k2, distortion.k3);
            const double radialDenominator = ComputeRadialDistortionScale(radiusSquared, distortion.k4, distortion.k5, distortion.k6);
            const double inverseRadialDenominator = Reciprocal(radialDenominator, 0.0);
            const double radialScale = (inverseRadialDenominator != 0.0) ? radialNumerator * inverseRadialDenominator : 1.0;

            const double twiceUndistortedXY = 2.0 * undistortedX * undistortedY;
            const double undistortedXSquared = Square(undistortedX);
            const double undistortedYSquared = Square(undistortedY);
            const double distortedX = undistortedX * radialScale + distortion.p1 * twiceUndistortedXY + distortion.p2 * (radiusSquared + 2.0 * undistortedXSquared);
            const double distortedY = undistortedY * radialScale + distortion.p1 * (radiusSquared + 2.0 * undistortedYSquared) + distortion.p2 * twiceUndistortedXY;

            const double pixelX = intrinsics.fx * distortedX + intrinsics.skew * distortedY + intrinsics.cx;
            const double pixelY = intrinsics.fy * distortedY + intrinsics.cy;

            if (!std::isfinite(pixelX) || !std::isfinite(pixelY))
            {
                dstRow[u] = 0U;
                continue;
            }

            const int pixelXFixed = static_cast<int>(std::lrint(pixelX * kInterpolationTabSize));
            const int pixelYFixed = static_cast<int>(std::lrint(pixelY * kInterpolationTabSize));
            const int x0 = pixelXFixed >> kInterpolationBits;
            const int y0 = pixelYFixed >> kInterpolationBits;
            const int fracX = pixelXFixed & kInterpolationTabMask;
            const int fracY = pixelYFixed & kInterpolationTabMask;

            const int v00 = samplePixel(x0, y0);
            const int v10 = samplePixel(x0 + 1, y0);
            const int v01 = samplePixel(x0, y0 + 1);
            const int v11 = samplePixel(x0 + 1, y0 + 1);

            const int w00 = (kInterpolationTabSize - fracX) * (kInterpolationTabSize - fracY);
            const int w10 = fracX * (kInterpolationTabSize - fracY);
            const int w01 = (kInterpolationTabSize - fracX) * fracY;
            const int w11 = fracX * fracY;

            const int weightedSum = v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11;
            const int interpolated = static_cast<int>(std::lrint(static_cast<double>(weightedSum) / kInterpolationTabSize2));
            const int clamped = std::clamp(interpolated, 0, 255);
            dstRow[u] = static_cast<std::uint8_t>(clamped);
        }
    }

    return Status{};
}

auto StereoRectify(const PinholeIntrinsics &intrinsics1, const DistortionCoefficients &distortion1, const PinholeIntrinsics &intrinsics2, const DistortionCoefficients &distortion2, const Mat3x3d &rotation, const Vec3d &translation, std::uint32_t imageWidth, std::uint32_t imageHeight, Mat3x3d &rectifiedRotation1, Mat3x3d &rectifiedRotation2, Mat3x4d &projectionMatrix1, Mat3x4d &projectionMatrix2, Mat4x4d &reprojectionMatrix, double alpha) noexcept -> Status
{
    if ((imageWidth == 0U) || (imageHeight == 0U))
    {
        LogError("imageWidth and imageHeight must be greater than zero.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    const Mat3x3d rectifyingRotation = ComputeInverseSqrtRotation(rotation);
    const Vec3d rectifyingTranslation = Multiply(rectifyingRotation, translation);
    const BaselineAxis baselineAxis = DetermineBaselineAxis(rectifyingTranslation);
    const Mat3x3d baselineAlignmentRotation = ComputeBaselineAxisAlignmentRotation(rectifyingTranslation, baselineAxis);
    const Mat3x3d inverseRectifyingRotation = Transpose(rectifyingRotation);
    rectifiedRotation1 = Multiply(baselineAlignmentRotation, inverseRectifyingRotation);
    rectifiedRotation2 = Multiply(baselineAlignmentRotation, rectifyingRotation);

    const Vec3d rectifiedTranslation = Multiply(rectifiedRotation2, translation);
    const double width = static_cast<double>(imageWidth);
    const double height = static_cast<double>(imageHeight);
    const double newFocalScale = 0.5;
    double newFocalLength = (baselineAxis == BaselineAxis::X) ? (intrinsics1.fy + intrinsics2.fy) * newFocalScale : (intrinsics1.fx + intrinsics2.fx) * newFocalScale;

    const Point2d principalPoint1 = ComputePrincipalPoint(intrinsics1, distortion1, rectifiedRotation1, newFocalLength, width, height);
    const Point2d principalPoint2 = ComputePrincipalPoint(intrinsics2, distortion2, rectifiedRotation2, newFocalLength, width, height);
    const Point2d principalPointAvg{0.5 * (principalPoint1[0] + principalPoint2[0]), 0.5 * (principalPoint1[1] + principalPoint2[1])};

    Point2d rectifiedPrincipalPoint1 = principalPointAvg;
    Point2d rectifiedPrincipalPoint2 = principalPointAvg;
    const double focalLengthScale = ComputeRectifiedFocalLengthScale(intrinsics1, distortion1, rectifiedRotation1, intrinsics2, distortion2, rectifiedRotation2, newFocalLength, rectifiedPrincipalPoint1, rectifiedPrincipalPoint2, imageWidth, imageHeight, alpha);
    newFocalLength *= focalLengthScale;

    projectionMatrix1 = ComputeProjectionMatrix(newFocalLength, rectifiedPrincipalPoint1);
    projectionMatrix2 = ComputeProjectionMatrix(newFocalLength, rectifiedPrincipalPoint2);

    const double baselineComponent = (baselineAxis == BaselineAxis::X) ? rectifiedTranslation[0] : rectifiedTranslation[1];
    const double translationOffset = baselineComponent * newFocalLength;
    if (baselineAxis == BaselineAxis::X)
    {
        projectionMatrix2[0][3] = translationOffset;
    }
    else
    {
        projectionMatrix2[1][3] = translationOffset;
    }

    reprojectionMatrix = Mat4x4d{};
    reprojectionMatrix[0][0] = 1.0;
    reprojectionMatrix[1][1] = 1.0;
    reprojectionMatrix[0][3] = -rectifiedPrincipalPoint1[0];
    reprojectionMatrix[1][3] = -rectifiedPrincipalPoint1[1];
    reprojectionMatrix[2][3] = newFocalLength;
    reprojectionMatrix[3][2] = (std::fabs(baselineComponent) > kEpsilon) ? (-1.0 / baselineComponent) : 0.0;
    reprojectionMatrix[3][3] = 0.0;

    return Status{};
}

auto InitUndistortRectifyMap(const PinholeIntrinsics &intrinsics, const DistortionCoefficients &distortion, const Mat3x3d &rotation, const Mat3x4d &projectionMatrix, std::uint32_t imageWidth, std::uint32_t imageHeight, float *mapX, std::size_t mapXStride, float *mapY, std::size_t mapYStride) noexcept -> Status
{
    if (mapX == nullptr || mapY == nullptr)
    {
        LogError("map pointers must not be null.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    if ((imageWidth == 0U) || (imageHeight == 0U))
    {
        LogError("imageWidth and imageHeight must be greater than zero.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    const std::size_t requiredStride = static_cast<std::size_t>(imageWidth) * sizeof(float);
    if (mapXStride < requiredStride)
    {
        LogError("mapX stride is too small for the given image width.");
        LogStrideError(mapXStride, requiredStride);
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    if (mapYStride < requiredStride)
    {
        LogError("mapY stride is too small for the given image width.");
        LogStrideError(mapYStride, requiredStride);
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    const Mat3x3d rotationInverse = Transpose(rotation);
    const auto &projectionRow0 = projectionMatrix[0];
    const auto &projectionRow1 = projectionMatrix[1];
    const double inverseRectifiedFocalX = Reciprocal(projectionRow0[0], 0.0);
    const double inverseRectifiedFocalY = Reciprocal(projectionRow1[1], 0.0);
    const double rectifiedPrincipalX = projectionRow0[2];
    const double rectifiedPrincipalY = projectionRow1[2];

    auto *mapXBytes = reinterpret_cast<unsigned char *>(mapX);
    auto *mapYBytes = reinterpret_cast<unsigned char *>(mapY);

    for (std::uint32_t v = 0; v < imageHeight; ++v)
    {
        const std::size_t offsetX = static_cast<std::size_t>(v) * mapXStride;
        const std::size_t offsetY = static_cast<std::size_t>(v) * mapYStride;
        auto *mapXRow = reinterpret_cast<float *>(mapXBytes + offsetX);
        auto *mapYRow = reinterpret_cast<float *>(mapYBytes + offsetY);
        const double rectifiedY = (static_cast<double>(v) - rectifiedPrincipalY) * inverseRectifiedFocalY;
        for (std::uint32_t u = 0; u < imageWidth; ++u)
        {
            const double rectifiedX = (static_cast<double>(u) - rectifiedPrincipalX) * inverseRectifiedFocalX;
            const Vec3d rectifiedPoint{rectifiedX, rectifiedY, 1.0};
            const Vec3d cameraPoint = Multiply(rotationInverse, rectifiedPoint);

            const double inverseDepth = Reciprocal(cameraPoint[2], 0.0);
            const double undistortedX = cameraPoint[0] * inverseDepth;
            const double undistortedY = cameraPoint[1] * inverseDepth;

            const double radiusSquared = Square(undistortedX) + Square(undistortedY);
            const double radialNumerator = ComputeRadialDistortionScale(radiusSquared, distortion.k1, distortion.k2, distortion.k3);
            const double radialDenominator = ComputeRadialDistortionScale(radiusSquared, distortion.k4, distortion.k5, distortion.k6);
            const double inverseRadialDenominator = Reciprocal(radialDenominator, 0.0);
            const double radialScale = (inverseRadialDenominator != 0.0) ? radialNumerator * inverseRadialDenominator : 1.0;

            const double twiceUndistortedXY = 2.0 * undistortedX * undistortedY;
            const double undistortedXSquared = Square(undistortedX);
            const double undistortedYSquared = Square(undistortedY);

            const double distortedX = undistortedX * radialScale + distortion.p1 * twiceUndistortedXY + distortion.p2 * (radiusSquared + 2.0 * undistortedXSquared);
            const double distortedY = undistortedY * radialScale + distortion.p1 * (radiusSquared + 2.0 * undistortedYSquared) + distortion.p2 * twiceUndistortedXY;

            const double pixelX = intrinsics.fx * distortedX + intrinsics.skew * distortedY + intrinsics.cx;
            const double pixelY = intrinsics.fy * distortedY + intrinsics.cy;
            mapXRow[u] = static_cast<float>(pixelX);
            mapYRow[u] = static_cast<float>(pixelY);
        }
    }

    return Status{};
}

auto InitIdentityMap(float *mapX, std::size_t mapXStride, float *mapY, std::size_t mapYStride, std::size_t imageWidth, std::size_t imageHeight) noexcept -> Status
{
    if (mapX == nullptr || mapY == nullptr)
    {
        LogError("map pointers must not be null.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    if ((imageWidth == 0U) || (imageHeight == 0U))
    {
        LogError("imageWidth and imageHeight must be greater than zero.");
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    const std::size_t requiredStride = static_cast<std::size_t>(imageWidth) * sizeof(float);
    if (mapXStride < requiredStride)
    {
        LogError("mapX stride is too small for the given image width.");
        LogStrideError(mapXStride, requiredStride);
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    if (mapYStride < requiredStride)
    {
        LogError("mapY stride is too small for the given image width.");
        LogStrideError(mapYStride, requiredStride);
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    auto *mapXBytes = reinterpret_cast<unsigned char *>(mapX);
    auto *mapYBytes = reinterpret_cast<unsigned char *>(mapY);

    for (std::size_t row = 0; row < imageHeight; ++row)
    {
        const std::size_t offsetX = row * mapXStride;
        const std::size_t offsetY = row * mapYStride;
        auto *mapXRow = reinterpret_cast<float *>(mapXBytes + offsetX);
        auto *mapYRow = reinterpret_cast<float *>(mapYBytes + offsetY);
        const float y = static_cast<float>(row);
        for (std::size_t col = 0; col < imageWidth; ++col)
        {
            mapXRow[col] = static_cast<float>(col);
            mapYRow[col] = y;
        }
    }

    return Status{};
}
} // namespace retinify
