// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/geometry.hpp"

#include <cmath>
#include <numbers>

namespace retinify
{
/// @brief Small epsilon constant for numerical stability.
constexpr double kEPS = 1e-12;

/// @brief Mathematical constant π (pi).
constexpr double kPI = 3.14159265358979323846;

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

auto Scale(const Vec3d &vec, double scale) noexcept -> Vec3d
{
    return {vec[0] * scale, vec[1] * scale, vec[2] * scale};
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
    if (n < kEPS)
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

    if (thetaSquared <= kEPS)
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
    if (theta < kEPS)
    {
        return {0.5 * vx, 0.5 * vy, 0.5 * vz};
    }

    // Near π: sinθ ~ 0, use diagonal-based axis extraction.
    if (std::fabs(kPI - theta) < 1e-6)
    {
        // n^2 from diagonal: n_x^2 = (R_xx + 1)/2, etc., clamped to [0,1].
        double ax = std::sqrt(std::max(0.0, (rotation[0][0] + 1.0) * 0.5));
        double ay = std::sqrt(std::max(0.0, (rotation[1][1] + 1.0) * 0.5));
        double az = std::sqrt(std::max(0.0, (rotation[2][2] + 1.0) * 0.5));

        // Recover remaining components from off-diagonals using the largest axis to stabilize.
        if (ax >= ay && ax >= az)
        {
            const double denom = 4.0 * std::max(ax, kEPS);
            ay = (rotation[0][1] + rotation[1][0]) / denom;
            az = (rotation[0][2] + rotation[2][0]) / denom;
        }
        else if (ay >= ax && ay >= az)
        {
            const double denom = 4.0 * std::max(ay, kEPS);
            ax = (rotation[0][1] + rotation[1][0]) / denom;
            az = (rotation[1][2] + rotation[2][1]) / denom;
        }
        else
        {
            const double denom = 4.0 * std::max(az, kEPS);
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
    if (vnorm < kEPS)
    {
        return {0.0, 0.0, 0.0};
    }
    const double scale = theta / vnorm;
    return {scale * vx, scale * vy, scale * vz};
}

auto UndistortPoint(const Intrinsics &intrinsics, const Distortion &distortion, const Point2d &pixel) noexcept -> Point2d
{
    const double invFocalX = (intrinsics.fx != 0.0 ? 1.0 / intrinsics.fx : 1.0);
    const double invFocalY = (intrinsics.fy != 0.0 ? 1.0 / intrinsics.fy : 1.0);

    const double normX = (pixel[0] - intrinsics.cx) * invFocalX;
    const double normY = (pixel[1] - intrinsics.cy) * invFocalY;

    double undistX = normX;
    double undistY = normY;

    std::array<double, 14> distCoeffs = {};
    distCoeffs[0] = distortion.k1; // k1
    distCoeffs[1] = distortion.k2; // k2
    distCoeffs[2] = distortion.p1; // p1
    distCoeffs[3] = distortion.p2; // p2
    distCoeffs[4] = distortion.k3; // k3
    distCoeffs[5] = distortion.k4; // k4
    distCoeffs[6] = distortion.k5; // k5
    distCoeffs[7] = distortion.k6; // k6
    distCoeffs[8] = 0.0;           // s1
    distCoeffs[9] = 0.0;           // s2
    distCoeffs[10] = 0.0;          // s3
    distCoeffs[11] = 0.0;          // s4
    distCoeffs[12] = 0.0;          // tau1
    distCoeffs[13] = 0.0;          // tau2

    constexpr int kIterationCount = 5;
    for (int iter = 0; iter < kIterationCount; ++iter)
    {
        double rSquared = undistX * undistX + undistY * undistY;
        double radialNum = 1.0 + ((distCoeffs[7] * rSquared + distCoeffs[6]) * rSquared + distCoeffs[5]) * rSquared;
        double radialDen = 1.0 + ((distCoeffs[4] * rSquared + distCoeffs[1]) * rSquared + distCoeffs[0]) * rSquared;
        double distScale = (std::fabs(radialDen) > kEPS) ? (radialNum / radialDen) : 1.0;
        if (distScale < 0)
        {
            undistX = normX;
            undistY = normY;
            break;
        }
        double deltaX = 2 * distCoeffs[2] * undistX * undistY + distCoeffs[3] * (rSquared + 2 * undistX * undistX) + distCoeffs[8] * rSquared + distCoeffs[9] * rSquared * rSquared;
        double deltaY = distCoeffs[2] * (rSquared + 2 * undistY * undistY) + 2 * distCoeffs[3] * undistX * undistY + distCoeffs[10] * rSquared + distCoeffs[11] * rSquared * rSquared;
        undistX = (normX - deltaX) * distScale;
        undistY = (normY - deltaY) * distScale;
    }
    return {undistX, undistY};
}

namespace impl
{
static auto ComputeRectifyingRotation(const Mat3x3d &rotation) noexcept -> Mat3x3d
{
    Vec3d omega = Log(rotation);
    for (double &component : omega)
    {
        component *= -0.5;
    }
    return Exp(omega);
}

static auto DetermineDominantAxis(const Vec3d &translation) noexcept -> int
{
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
    const double component = translation[axisIdx];
    const double length = Length(translation);
    const Vec3d target = BuildAxisVector(axisIdx, component > 0.0 ? 1.0 : -1.0);
    const Vec3d cross = Cross(translation, target);
    const double crossLength = Length(cross);
    if (crossLength <= 0.0 || length <= 0.0)
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
    const Point2d corners[4] = {{0.0, 0.0}, {width - 1.0, 0.0}, {0.0, height - 1.0}, {width - 1.0, height - 1.0}};
    double accumulatedX = 0.0;
    double accumulatedY = 0.0;
    for (const auto &corner : corners)
    {
        const Point2d undistortedPoint2D = UndistortPoint(intrinsics, distortion, corner);
        const Vec3d undistortedPoint3D{undistortedPoint2D[0], undistortedPoint2D[1], 1.0};
        const Vec3d rectifiedPoint3D = Multiply(rectifiedRotation, undistortedPoint3D);
        const double inverseDepth = (std::fabs(rectifiedPoint3D[2]) > kEPS) ? (1.0 / rectifiedPoint3D[2]) : 1.0;
        accumulatedX += newFocalLength * (rectifiedPoint3D[0] * inverseDepth);
        accumulatedY += newFocalLength * (rectifiedPoint3D[1] * inverseDepth);
    }
    const double halfWidth = (width - 1.0) * 0.5;
    const double halfHeight = (height - 1.0) * 0.5;
    return {halfWidth - accumulatedX * 0.25, halfHeight - accumulatedY * 0.25};
}

static auto BuildProjectionMatrix(double focal, const Point2d &principal) noexcept -> Mat3x4d
{
    Mat3x4d projection = Mat3x4d{};
    projection[0][0] = focal;
    projection[1][1] = focal;
    projection[0][2] = principal[0];
    projection[1][2] = principal[1];
    projection[2][2] = 1.0;
    return projection;
}
} // namespace impl

auto StereoRectify(const Intrinsics &K1, const Distortion &D1, //
                   const Intrinsics &K2, const Distortion &D2, //
                   int width, int height,                      //
                   const Mat3x3d &R, const Vec3d &T,           //
                   Mat3x3d &R1, Mat3x3d &R2,                   //
                   Mat3x4d &P1, Mat3x4d &P2,                   //
                   Mat4x4d &Q) noexcept -> void
{
    const Mat3x3d rectifyingRotation = impl::ComputeRectifyingRotation(R);
    const Vec3d rotatedTranslation = Multiply(rectifyingRotation, T);
    const int axisIdx = impl::DetermineDominantAxis(rotatedTranslation);
    const Mat3x3d baselineAlignment = impl::ComputeBaselineAlignment(rotatedTranslation, axisIdx);

    const Mat3x3d rotationTranspose = Transpose(rectifyingRotation);
    R1 = Multiply(baselineAlignment, rotationTranspose);
    R2 = Multiply(baselineAlignment, rectifyingRotation);

    const Vec3d rectifiedTranslation = Multiply(R2, T);
    const double newFocalScale = 0.5; // newImgSize defaults to imageSize -> ratio = 0.5
    const double newFocalLength = (axisIdx == 0) ? (K1.fy + K2.fy) * newFocalScale : (K1.fx + K2.fx) * newFocalScale;

    const Point2d principal0 = impl::ComputePrincipalPoint(K1, D1, R1, newFocalLength, static_cast<double>(width), static_cast<double>(height));
    const Point2d principal1 = impl::ComputePrincipalPoint(K2, D2, R2, newFocalLength, static_cast<double>(width), static_cast<double>(height));
    const double ccx = 0.5 * (principal0[0] + principal1[0]);
    const double ccy = 0.5 * (principal0[1] + principal1[1]);
    const Point2d principalAvg{ccx, ccy};

    P1 = impl::BuildProjectionMatrix(newFocalLength, principalAvg);
    P2 = impl::BuildProjectionMatrix(newFocalLength, principalAvg);

    const double baselineComponent = (axisIdx == 0) ? rectifiedTranslation[0] : rectifiedTranslation[1];
    const double translationOffset = baselineComponent * newFocalLength;
    if (axisIdx == 0)
    {
        P2[0][3] = translationOffset;
    }
    else
    {
        P2[1][3] = translationOffset;
    }

    Q = Mat4x4d{};
    Q[0][0] = 1.0;
    Q[1][1] = 1.0;
    Q[0][3] = -principalAvg[0];
    Q[1][3] = -principalAvg[1];
    Q[2][3] = newFocalLength;
    Q[3][2] = (std::fabs(baselineComponent) > kEPS) ? (-1.0 / baselineComponent) : 0.0;
    Q[3][3] = 0.0;
}
} // namespace retinify
