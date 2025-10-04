// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace retinify
{
namespace
{
constexpr inline double kEpsilon = 1e-12;
constexpr inline double kPi = 3.141592653589793;
constexpr inline std::size_t kMatSize = 3;

[[nodiscard]] constexpr double Square(double value) noexcept
{
    return value * value;
}

[[nodiscard]] constexpr double Dot(const Vec3d &lhs, const Vec3d &rhs) noexcept
{
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

[[nodiscard]] constexpr Mat3x3d ComputeSkewSymmetricMatrix(const Vec3d &omega) noexcept
{
    return {{{0.0, -omega[2], omega[1]}, //
             {omega[2], 0.0, -omega[0]}, //
             {-omega[1], omega[0], 0.0}}};
}

inline void AccumulateScaledMatrix(Mat3x3d &target, const Mat3x3d &addend, double scale) noexcept
{
    for (std::size_t row = 0; row < kMatSize; ++row)
    {
        for (std::size_t col = 0; col < kMatSize; ++col)
        {
            target[row][col] += addend[row][col] * scale;
        }
    }
}

[[nodiscard]] constexpr bool IsBorderIndex(int index, int lastIndex) noexcept
{
    return index == 0 || index == lastIndex;
}

[[nodiscard]] constexpr double EvaluateRadialPolynomial(double radiusSquared, double k1, double k2, double k3) noexcept
{
    const double radiusFourth = radiusSquared * radiusSquared;
    const double radiusSixth = radiusFourth * radiusSquared;
    return 1.0 + k1 * radiusSquared + k2 * radiusFourth + k3 * radiusSixth;
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
    for (std::size_t row = 0; row < kMatSize; ++row)
    {
        for (std::size_t col = 0; col < kMatSize; ++col)
        {
            transposed[row][col] = mat[col][row];
        }
    }
    return transposed;
}

auto Multiply(const Mat3x3d &mat, const Vec3d &vec) noexcept -> Vec3d
{
    // y = R x
    Vec3d result{};
    for (std::size_t row = 0; row < kMatSize; ++row)
    {
        const auto &matRow = mat[row];
        result[row] = matRow[0] * vec[0] + matRow[1] * vec[1] + matRow[2] * vec[2];
    }
    return result;
}

auto Multiply(const Mat3x3d &mat1, const Mat3x3d &mat2) noexcept -> Mat3x3d
{
    // C = A B,  c_{ij} = Σ_k a_{ik} b_{kj}
    Mat3x3d result{};
    for (std::size_t row = 0; row < kMatSize; ++row)
    {
        const auto &rowValues = mat1[row];
        for (std::size_t col = 0; col < kMatSize; ++col)
        {
            result[row][col] = rowValues[0] * mat2[0][col] + rowValues[1] * mat2[1][col] + rowValues[2] * mat2[2][col];
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
    return std::sqrt(Dot(vec, vec));
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
    const double thetaSquared = Dot(omega, omega);
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
    const Mat3x3d skew = ComputeSkewSymmetricMatrix(omega);
    const Mat3x3d skewSquared = Multiply(skew, skew);

    Mat3x3d rotation = Identity();
    AccumulateScaledMatrix(rotation, skew, coefA);
    AccumulateScaledMatrix(rotation, skewSquared, coefB);
    return rotation;
}

auto Log(const Mat3x3d &rotation) noexcept -> Vec3d
{
    // θ from trace: cosθ = (tr(R) − 1)/2
    const double trace = rotation[0][0] + rotation[1][1] + rotation[2][2];
    const double cosTheta = std::clamp((trace - 1.0) * 0.5, -1.0, 1.0);
    const double theta = std::acos(cosTheta);

    // v = (R − R^T)∨ = 2 sinθ · n  (skew vector)
    const Vec3d skewVector{rotation[2][1] - rotation[1][2], rotation[0][2] - rotation[2][0], rotation[1][0] - rotation[0][1]};

    // Small angle: ω ≈ 1/2 v (since sinθ ≈ θ)
    if (theta < kEpsilon)
    {
        return Scale(skewVector, 0.5);
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
        Vec3d axisCandidate{ax, ay, az};
        Vec3d axis = Normalize(axisCandidate);
        return Scale(axis, theta);
    }

    // General case: n = v / ||v||, ω = θ n
    const double skewVectorNormSquared = Dot(skewVector, skewVector);
    const double skewVectorNorm = std::sqrt(std::max(0.0, skewVectorNormSquared));
    if (skewVectorNorm < kEpsilon)
    {
        return {0.0, 0.0, 0.0};
    }
    const double scale = theta / skewVectorNorm;
    return Scale(skewVector, scale);
}

auto UndistortPoint(const Intrinsics &intrinsics, const Distortion &distortion, const Point2d &pixel) noexcept -> Point2d
{
    const double invFocalX = Reciprocal(intrinsics.fx, 1.0);
    const double invFocalY = Reciprocal(intrinsics.fy, 1.0);

    const double normalizedX = (pixel[0] - intrinsics.cx) * invFocalX;
    const double normalizedY = (pixel[1] - intrinsics.cy) * invFocalY;

    double undistortedX = normalizedX;
    double undistortedY = normalizedY;

    constexpr int kIterationCount = 5;
    for (int iter = 0; iter < kIterationCount; ++iter)
    {
        // Brown-Conrady rational model inversion with r^2 = x^2 + y^2:
        // radialScale = (1 + k4 r^2 + k5 r^4 + k6 r^6) / (1 + k1 r^2 + k2 r^4 + k3 r^6)
        // Delta tangential = (2 p1 xy + p2 (r^2 + 2 x^2),
        //                     p1 (r^2 + 2 y^2) + 2 p2 xy)
        const double radiusSquared = Square(undistortedX) + Square(undistortedY);
        const double radialNumerator = EvaluateRadialPolynomial(radiusSquared, distortion.k4, distortion.k5, distortion.k6);
        const double radialDenominator = EvaluateRadialPolynomial(radiusSquared, distortion.k1, distortion.k2, distortion.k3);
        const double invRadialDenominator = Reciprocal(radialDenominator, 0.0);
        const double radialScale = (invRadialDenominator != 0.0) ? radialNumerator * invRadialDenominator : 1.0;
        if (radialScale < 0.0)
        {
            undistortedX = normalizedX;
            undistortedY = normalizedY;
            break;
        }

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

namespace
{
[[nodiscard]] static auto ComputeRectifyingRotation(const Mat3x3d &rotation) noexcept -> Mat3x3d
{
    // Split relative rotation equally: R_rect = exp(-0.5 * log(R))
    const Vec3d omega = Log(rotation);
    return Exp(Scale(omega, -0.5));
}

// Orientation of the rectified stereo baseline.
enum class BaselineAxis : std::uint8_t
{
    X = 0,
    Y = 1
};

[[nodiscard]] constexpr auto ToAxisIndex(BaselineAxis axis) noexcept -> std::uint8_t
{
    return static_cast<std::uint8_t>(axis);
}

[[nodiscard]] static auto DetermineDominantAxis(const Vec3d &translation) noexcept -> BaselineAxis
{
    // argmax_i |T_i|
    return (std::fabs(translation[0]) > std::fabs(translation[1])) ? BaselineAxis::X : BaselineAxis::Y;
}

[[nodiscard]] static auto ComputeAxisVector(BaselineAxis axis, double direction) noexcept -> Vec3d
{
    Vec3d axisVector{0.0, 0.0, 0.0};
    axisVector[ToAxisIndex(axis)] = direction;
    return axisVector;
}

[[nodiscard]] static auto ComputeBaselineAlignment(const Vec3d &translation, BaselineAxis axis) noexcept -> Mat3x3d
{
    // Rotate translation onto +/- e_axis using axis = T x target, angle = acos(|T_axis| / ||T||)
    const int axisIndex = ToAxisIndex(axis);
    const double component = translation[axisIndex];
    const double length = Length(translation);
    const Vec3d targetAxisVector = ComputeAxisVector(axis, component >= 0.0 ? 1.0 : -1.0);
    const Vec3d cross = Cross(translation, targetAxisVector);
    const double crossLength = Length(cross);
    if (crossLength <= kEpsilon || length <= kEpsilon)
    {
        return Identity();
    }
    const double arg = std::clamp(std::fabs(component) / length, -1.0, 1.0);
    const double angle = std::acos(arg);
    const double scale = angle / crossLength;
    return Exp(Scale(cross, scale));
}

static auto ComputePrincipalPoint(const Intrinsics &intrinsics, const Distortion &distortion, const Mat3x3d &rectifiedRotation, //
                                  double newFocalLength, double width, double height) noexcept -> Point2d
{
    // Principal shift: c = (w-1, h-1)/2 - f * average( rectified_xy / rectified_z ) over undistorted corners
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

[[nodiscard]] static auto ComputeCameraMatrix(double focalLength, const Point2d &principalPoint) noexcept -> Mat3x3d
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

[[nodiscard]] double ComputeSafeRatio(double numerator, double denominator) noexcept
{
    if (std::fabs(denominator) <= kEpsilon)
    {
        if (std::fabs(numerator) <= kEpsilon)
        {
            return 0.0;
        }
        const double sign = (numerator >= 0.0) ? 1.0 : -1.0;
        return sign * std::numeric_limits<double>::infinity();
    }
    return numerator / denominator;
}

static void ComputeUndistortRectangles(const Intrinsics &intrinsics, const Distortion &distortion,           //
                                       const Mat3x3d &rectificationRotation, const Mat3x3d &newCameraMatrix, //
                                       std::uint32_t imageWidth, std::uint32_t imageHeight,                  //
                                       Rect2d &inner, Rect2d &outer) noexcept
{
    constexpr int kGridSize = 9;
    const int lastIndex = kGridSize - 1;
    const double width = static_cast<double>(imageWidth);
    const double height = static_cast<double>(imageHeight);
    const double maxX = width - 1.0;
    const double maxY = height - 1.0;
    const double stepX = (kGridSize > 1) ? (maxX / static_cast<double>(lastIndex)) : 0.0;
    const double stepY = (kGridSize > 1) ? (maxY / static_cast<double>(lastIndex)) : 0.0;

    double innerMinX = -std::numeric_limits<double>::infinity();
    double innerMaxX = std::numeric_limits<double>::infinity();
    double innerMinY = -std::numeric_limits<double>::infinity();
    double innerMaxY = std::numeric_limits<double>::infinity();
    double outerMinX = std::numeric_limits<double>::infinity();
    double outerMaxX = -std::numeric_limits<double>::infinity();
    double outerMinY = std::numeric_limits<double>::infinity();
    double outerMaxY = -std::numeric_limits<double>::infinity();

    const auto &row0 = newCameraMatrix[0];
    const auto &row1 = newCameraMatrix[1];
    const auto &row2 = newCameraMatrix[2];

    const auto projectRectifiedPoint = [&](double rectifiedX, double rectifiedY) noexcept -> Point2d {
        const double numeratorX = row0[0] * rectifiedX + row0[1] * rectifiedY + row0[2];
        const double numeratorY = row1[0] * rectifiedX + row1[1] * rectifiedY + row1[2];
        const double denominator = row2[0] * rectifiedX + row2[1] * rectifiedY + row2[2];
        const double invDenominator = Reciprocal(denominator, 0.0);
        const double scale = (invDenominator == 0.0) ? 1.0 : invDenominator;
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
        const bool onYBorder = IsBorderIndex(gridY, lastIndex);
        const bool onTopEdge = (gridY == 0);
        const bool onBottomEdge = (gridY == lastIndex);
        const double pixelY = stepY * static_cast<double>(gridY);

        for (int gridX = 0; gridX < kGridSize; ++gridX)
        {
            const bool onXBorder = IsBorderIndex(gridX, lastIndex);
            if (!onXBorder && !onYBorder)
            {
                continue;
            }

            const double pixelX = stepX * static_cast<double>(gridX);
            const Point2d undistorted = UndistortPoint(intrinsics, distortion, {pixelX, pixelY});
            const Vec3d undistorted3D{undistorted[0], undistorted[1], 1.0};
            const Vec3d rectified = Multiply(rectificationRotation, undistorted3D);
            const double invDepth = Reciprocal(rectified[2], 0.0);
            const double rectifiedX = rectified[0] * invDepth;
            const double rectifiedY = rectified[1] * invDepth;

            const Point2d mappedPoint = projectRectifiedPoint(rectifiedX, rectifiedY);
            const bool onLeftEdge = (gridX == 0);
            const bool onRightEdge = (gridX == lastIndex);
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

[[nodiscard]] static auto ComputeAlphaScale(double alpha,                                                                                        //
                                            const Intrinsics &intrinsics1, const Distortion &distortion1, const Mat3x3d &rectificationRotation1, //
                                            const Intrinsics &intrinsics2, const Distortion &distortion2, const Mat3x3d &rectificationRotation2, //
                                            double focalLength, const Point2d &principal1, const Point2d &principal2,                            //
                                            std::uint32_t imageWidth, std::uint32_t imageHeight) noexcept -> double
{
    if (alpha < 0.0)
    {
        return 1.0;
    }

    const double clampedAlpha = std::clamp(alpha, 0.0, 1.0);

    const Mat3x3d cameraMatrix1 = ComputeCameraMatrix(focalLength, principal1);
    const Mat3x3d cameraMatrix2 = ComputeCameraMatrix(focalLength, principal2);

    Rect2d inner1{};
    Rect2d outer1{};
    Rect2d inner2{};
    Rect2d outer2{};
    ComputeUndistortRectangles(intrinsics1, distortion1, rectificationRotation1, cameraMatrix1, imageWidth, imageHeight, inner1, outer1);
    ComputeUndistortRectangles(intrinsics2, distortion2, rectificationRotation2, cameraMatrix2, imageWidth, imageHeight, inner2, outer2);

    const double width = static_cast<double>(imageWidth);
    const double height = static_cast<double>(imageHeight);

    const auto computeInnerScale = [&](const Rect2d &inner, const Point2d &principal) noexcept -> double {
        const double cx = principal[0];
        const double cy = principal[1];
        const double scaleLeft = ComputeSafeRatio(cx, cx - inner.x);
        const double scaleRight = ComputeSafeRatio(width - 1.0 - cx, inner.x + inner.width - cx);
        const double scaleTop = ComputeSafeRatio(cy, cy - inner.y);
        const double scaleBottom = ComputeSafeRatio(height - 1.0 - cy, inner.y + inner.height - cy);
        return std::max(std::max(scaleLeft, scaleRight), std::max(scaleTop, scaleBottom));
    };

    const auto computeOuterScale = [&](const Rect2d &outer, const Point2d &principal) noexcept -> double {
        const double cx = principal[0];
        const double cy = principal[1];
        const double scaleLeft = ComputeSafeRatio(cx, cx - outer.x);
        const double scaleRight = ComputeSafeRatio(width - 1.0 - cx, outer.x + outer.width - cx);
        const double scaleTop = ComputeSafeRatio(cy, cy - outer.y);
        const double scaleBottom = ComputeSafeRatio(height - 1.0 - cy, outer.y + outer.height - cy);
        return std::min(std::min(scaleLeft, scaleRight), std::min(scaleTop, scaleBottom));
    };

    const double innerScale1 = computeInnerScale(inner1, principal1);
    const double innerScale2 = computeInnerScale(inner2, principal2);
    const double s0 = std::max(innerScale1, innerScale2);

    const double outerScale1 = computeOuterScale(outer1, principal1);
    const double outerScale2 = computeOuterScale(outer2, principal2);
    const double s1 = std::min(outerScale1, outerScale2);

    const double scale = s0 * (1.0 - clampedAlpha) + s1 * clampedAlpha;
    if (!std::isfinite(scale) || scale <= 0.0)
    {
        return 1.0;
    }
    return scale;
}

static auto ComputeProjectionMatrix(double focal, const Point2d &principal) noexcept -> Mat3x4d
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
                   Mat4x4d &mappingMatrix, double alpha) noexcept -> void
{
    // Stereo rectification: rotation1 = R_align R_rect^T, rotation2 = R_align R_rect,
    // share focal f = 0.5 * ((fx1+fx2) if axis=y else (fy1+fy2)), and set disparity scale via baseline
    const Mat3x3d rectifyingRotation = ComputeRectifyingRotation(rotation);
    const Vec3d rotatedTranslation = Multiply(rectifyingRotation, translation);
    const BaselineAxis dominantAxis = DetermineDominantAxis(rotatedTranslation);
    const Mat3x3d baselineAlignment = ComputeBaselineAlignment(rotatedTranslation, dominantAxis);

    const Mat3x3d rotationTranspose = Transpose(rectifyingRotation);
    rotation1 = Multiply(baselineAlignment, rotationTranspose);
    rotation2 = Multiply(baselineAlignment, rectifyingRotation);

    const Vec3d rectifiedTranslation = Multiply(rotation2, translation);
    const double width = static_cast<double>(imageWidth);
    const double height = static_cast<double>(imageHeight);
    const double newFocalScale = 0.5;
    double newFocalLength = (dominantAxis == BaselineAxis::X) ? (intrinsics1.fy + intrinsics2.fy) * newFocalScale : (intrinsics1.fx + intrinsics2.fx) * newFocalScale;

    const Point2d principal1 = ComputePrincipalPoint(intrinsics1, distortion1, rotation1, newFocalLength, width, height);
    const Point2d principal2 = ComputePrincipalPoint(intrinsics2, distortion2, rotation2, newFocalLength, width, height);
    const Point2d principalAvg{0.5 * (principal1[0] + principal2[0]), 0.5 * (principal1[1] + principal2[1])};

    Point2d rectifiedPrincipal1 = principalAvg;
    Point2d rectifiedPrincipal2 = principalAvg;
    const double alphaScale = ComputeAlphaScale(alpha, intrinsics1, distortion1, rotation1, intrinsics2, distortion2, rotation2, newFocalLength, rectifiedPrincipal1, rectifiedPrincipal2, imageWidth, imageHeight);
    newFocalLength *= alphaScale;

    projectionMatrix1 = ComputeProjectionMatrix(newFocalLength, rectifiedPrincipal1);
    projectionMatrix2 = ComputeProjectionMatrix(newFocalLength, rectifiedPrincipal2);

    const double baselineComponent = (dominantAxis == BaselineAxis::X) ? rectifiedTranslation[0] : rectifiedTranslation[1];
    const double translationOffset = baselineComponent * newFocalLength;
    if (dominantAxis == BaselineAxis::X)
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
    mappingMatrix[0][3] = -rectifiedPrincipal1[0];
    mappingMatrix[1][3] = -rectifiedPrincipal1[1];
    mappingMatrix[2][3] = newFocalLength;
    mappingMatrix[3][2] = (std::fabs(baselineComponent) > kEpsilon) ? (-1.0 / baselineComponent) : 0.0;
    mappingMatrix[3][3] = 0.0;
}

auto InitUndistortRectifyMap(const Intrinsics &intrinsics, const Distortion &distortion, //
                             const Mat3x3d &rotation, const Mat3x4d &projectionMatrix,   //
                             std::uint32_t imageWidth, std::uint32_t imageHeight,        //
                             float *mapX, std::size_t mapXStride,                        //
                             float *mapY, std::size_t mapYStride) noexcept -> void
{
    if (mapX == nullptr || mapY == nullptr)
    {
        return;
    }

    const Mat3x3d rotationInverse = Transpose(rotation);
    const auto &projectionRow0 = projectionMatrix[0];
    const auto &projectionRow1 = projectionMatrix[1];
    const double invRectifiedFocalX = Reciprocal(projectionRow0[0], 0.0);
    const double invRectifiedFocalY = Reciprocal(projectionRow1[1], 0.0);
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
            const double radialNumerator = EvaluateRadialPolynomial(radiusSquared, distortion.k1, distortion.k2, distortion.k3);
            const double radialDenominator = EvaluateRadialPolynomial(radiusSquared, distortion.k4, distortion.k5, distortion.k6);
            const double invRadialDenominator = Reciprocal(radialDenominator, 0.0);
            const double radialScale = (invRadialDenominator != 0.0) ? radialNumerator * invRadialDenominator : 1.0;

            const double twoNormalizedXY = 2.0 * normalizedX * normalizedY;
            const double normalizedXSquared = Square(normalizedX);
            const double normalizedYSquared = Square(normalizedY);

            const double distortedNormalizedX = normalizedX * radialScale + distortion.p1 * twoNormalizedXY + distortion.p2 * (radiusSquared + 2.0 * normalizedXSquared);
            const double distortedNormalizedY = normalizedY * radialScale + distortion.p1 * (radiusSquared + 2.0 * normalizedYSquared) + distortion.p2 * twoNormalizedXY;

            const double uDistorted = intrinsics.fx * distortedNormalizedX + intrinsics.skew * distortedNormalizedY + intrinsics.cx;
            const double vDistorted = intrinsics.fy * distortedNormalizedY + intrinsics.cy;
            mapXRow[u] = static_cast<float>(uDistorted);
            mapYRow[u] = static_cast<float>(vDistorted);
        }
    }
}

auto InitIdentityMap(float *mapX, std::size_t mapXStride, //
                     float *mapY, std::size_t mapYStride, //
                     std::size_t imageWidth, std::size_t imageHeight) noexcept -> void
{
    if (mapX == nullptr || mapY == nullptr)
    {
        return;
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
}
} // namespace retinify
