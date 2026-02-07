// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "attributes.hpp"
#include "status.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace retinify
{
/// @brief
/// 2D vector (double)
using Vec2d = std::array<double, 2>;

/// @brief
/// 3D vector (double)
using Vec3d = std::array<double, 3>;

/// @brief
/// 2D point (double)
using Point2d = std::array<double, 2>;

/// @brief
/// 3D point (double)
using Point3d = std::array<double, 3>;

/// @brief
/// 3x3 matrix (double, row-major)
using Mat3x3d = std::array<std::array<double, 3>, 3>;

/// @brief
/// 3x4 matrix (double, row-major)
using Mat3x4d = std::array<std::array<double, 4>, 3>;

/// @brief
/// 4x4 matrix (double, row-major)
using Mat4x4d = std::array<std::array<double, 4>, 4>;

/// @brief
/// Rectangle structure
/// @tparam T
/// Type of the rectangle coordinates and dimensions
template <typename T> struct Rect2
{
    /// @brief
    /// X coordinate of the top-left corner
    T x{0};
    /// @brief
    /// Y coordinate of the top-left corner
    T y{0};
    /// @brief
    /// Width of the rectangle
    T width{0};
    /// @brief
    /// Height of the rectangle
    T height{0};
};

/// @brief
/// 2D rectangle (double)
using Rect2d = Rect2<double>;

/// @brief
/// Create a 3x3 identity matrix
/// @return
/// 3x3 identity matrix
RETINIFY_API auto Identity() noexcept -> Mat3x3d;

/// @brief
/// Compute the determinant of a 3x3 matrix
/// @param mat
/// 3x3 matrix
/// @return
/// Determinant value
RETINIFY_API auto Determinant(const Mat3x3d &mat) noexcept -> double;

/// @brief
/// Transpose a 3x3 matrix
/// @param mat
/// 3x3 matrix
/// @return
/// Transposed 3x3 matrix
RETINIFY_API auto Transpose(const Mat3x3d &mat) noexcept -> Mat3x3d;

/// @brief
/// Add two 3x3 matrices
/// @param mat1
/// First 3x3 matrix
/// @param mat2
/// Second 3x3 matrix
/// @return
/// 3x3 matrix
RETINIFY_API auto Add(const Mat3x3d &mat1, const Mat3x3d &mat2) noexcept -> Mat3x3d;

/// @brief
/// Multiply a 3x3 matrix and a 3D vector
/// @param mat
/// 3x3 matrix
/// @param vec
/// 3D vector
/// @return
/// 3D vector
RETINIFY_API auto Multiply(const Mat3x3d &mat, const Vec3d &vec) noexcept -> Vec3d;

/// @brief
/// Multiply two 3x3 matrices
/// @param mat1
/// First 3x3 matrix
/// @param mat2
/// Second 3x3 matrix
/// @return
/// 3x3 matrix
RETINIFY_API auto Multiply(const Mat3x3d &mat1, const Mat3x3d &mat2) noexcept -> Mat3x3d;

/// @brief
/// Multiply a 3x3 matrix by a scalar value
/// @param mat
/// 3x3 matrix
/// @param scale
/// Scalar value
/// @return
/// 3x3 matrix
RETINIFY_API auto Multiply(const Mat3x3d &mat, double scale) noexcept -> Mat3x3d;

/// @brief
/// Multiply a 3D vector by a scalar value
/// @param vec
/// 3D vector
/// @param scale
/// Scalar value
/// @return
/// 3D vector
RETINIFY_API auto Multiply(const Vec3d &vec, double scale) noexcept -> Vec3d;

/// @brief
/// Compute the length (magnitude) of a 3D vector
/// @param vec
/// 3D vector
/// @return
/// Length value
RETINIFY_API auto Length(const Vec3d &vec) noexcept -> double;

/// @brief
/// Normalize a 3D vector to unit length
/// @param vec
/// 3D vector
/// @return
/// Normalized 3D vector
RETINIFY_API auto Normalize(const Vec3d &vec) noexcept -> Vec3d;

/// @brief
/// Compute the dot product of two 3D vectors
/// @param vec1
/// First 3D vector
/// @param vec2
/// Second 3D vector
/// @return
/// Dot product value
RETINIFY_API auto Dot(const Vec3d &vec1, const Vec3d &vec2) noexcept -> double;

/// @brief
/// Compute the cross product of two 3D vectors
/// @param vec1
/// First 3D vector
/// @param vec2
/// Second 3D vector
/// @return
/// Cross product vector
RETINIFY_API auto Cross(const Vec3d &vec1, const Vec3d &vec2) noexcept -> Vec3d;

/// @brief
/// Create a 3x3 skew-symmetric matrix from a 3D rotation vector
/// @param vec
/// 3D rotation vector
/// @return
/// 3x3 skew-symmetric matrix
RETINIFY_API auto Hat(const Vec3d &vec) noexcept -> Mat3x3d;

/// @brief
/// Convert a 3x3 skew-symmetric matrix to a 3D rotation vector
/// @param mat
/// 3x3 skew-symmetric matrix
/// @return
/// 3D rotation vector
RETINIFY_API auto Vee(const Mat3x3d &mat) noexcept -> Vec3d;

/// @brief
/// Compute the matrix exponential of a 3D rotation vector
/// @param vec
/// 3D rotation vector
/// @return
/// 3x3 rotation matrix
RETINIFY_API auto Exp(const Vec3d &vec) noexcept -> Mat3x3d;

/// @brief
/// Compute the matrix logarithm of a 3x3 rotation matrix
/// @param mat
/// 3x3 rotation matrix
/// @return
/// 3D rotation vector
RETINIFY_API auto Log(const Mat3x3d &mat) noexcept -> Vec3d;

/// @brief
/// Pinhole camera intrinsic parameters with focal lengths, principal point, and skew
struct PinholeIntrinsics
{
    /// @brief
    /// Focal length in x (in pixels)
    double fx{0};
    /// @brief
    /// Focal length in y (in pixels)
    double fy{0};
    /// @brief
    /// Principal point x-coordinate (in pixels)
    double cx{0};
    /// @brief
    /// Principal point y-coordinate (in pixels)
    double cy{0};
    /// @brief
    /// Skew coefficient
    double skew{0};

    [[nodiscard]] auto operator==(const PinholeIntrinsics &other) const noexcept -> bool
    {
        return fx == other.fx && //
               fy == other.fy && //
               cx == other.cx && //
               cy == other.cy && //
               skew == other.skew;
    }
};

/// @brief
/// Rational distortion model with 8 coefficients: (k1, k2, p1, p2, k3, k4, k5, k6)
struct DistortionCoefficients
{
    double k1{0};
    double k2{0};
    double p1{0};
    double p2{0};
    double k3{0};
    double k4{0};
    double k5{0};
    double k6{0};

    [[nodiscard]] auto operator==(const DistortionCoefficients &other) const noexcept -> bool
    {
        return k1 == other.k1 && //
               k2 == other.k2 && //
               p1 == other.p1 && //
               p2 == other.p2 && //
               k3 == other.k3 && //
               k4 == other.k4 && //
               k5 == other.k5 && //
               k6 == other.k6;
    }
};

/// @brief
/// Stereo camera calibration parameters
struct CalibrationParameters
{
    /// @brief
    /// Pinhole intrinsics for the left camera
    PinholeIntrinsics leftIntrinsics{};
    /// @brief
    /// Distortion coefficients for the left camera
    DistortionCoefficients leftDistortion{};
    /// @brief
    /// Pinhole intrinsics for the right camera
    PinholeIntrinsics rightIntrinsics{};
    /// @brief
    /// Distortion coefficients for the right camera
    DistortionCoefficients rightDistortion{};
    /// @brief
    /// Rotation matrix
    Mat3x3d rotation{};
    /// @brief
    /// Translation vector
    Vec3d translation{};
    /// @brief
    /// Image width (in pixels)
    std::uint32_t imageWidth{};
    /// @brief
    /// Image height (in pixels)
    std::uint32_t imageHeight{};
    /// @brief
    /// Root mean square reprojection error (in pixels)
    double calibrationError{};
    /// @brief
    /// Calibration timestamp (in seconds since epoch, UTC)
    std::int64_t calibrationTime{};

    [[nodiscard]] auto operator==(const CalibrationParameters &other) const noexcept -> bool
    {
        return leftIntrinsics == other.leftIntrinsics &&     //
               leftDistortion == other.leftDistortion &&     //
               rightIntrinsics == other.rightIntrinsics &&   //
               rightDistortion == other.rightDistortion &&   //
               rotation == other.rotation &&                 //
               translation == other.translation &&           //
               imageWidth == other.imageWidth &&             //
               imageHeight == other.imageHeight &&           //
               calibrationError == other.calibrationError && //
               calibrationTime == other.calibrationTime;     //
    }
};

/// @brief
/// Undistort a 2D point using the given camera intrinsics and distortion coefficients
/// @param intrinsics
/// Camera intrinsic parameters
/// @param distortion
/// Distortion coefficients
/// @param point
/// Distorted 2D point (in pixel coordinates)
/// @return
/// Undistorted 2D point (normalized image coordinates)
RETINIFY_API auto UndistortPoint(const PinholeIntrinsics &intrinsics, const DistortionCoefficients &distortion, const Point2d &point) noexcept -> Point2d;

/// @brief
/// Distort a normalized 2D point using the given camera intrinsics and distortion coefficients
/// @param intrinsics
/// Camera intrinsic parameters
/// @param distortion
/// Distortion coefficients
/// @param point
/// Undistorted 2D point (normalized image coordinates)
/// @return
/// Distorted 2D point (in pixel coordinates)
RETINIFY_API auto DistortPoint(const PinholeIntrinsics &intrinsics, const DistortionCoefficients &distortion, const Point2d &point) noexcept -> Point2d;

/// @brief
/// Undistort an image using the given camera intrinsics and distortion coefficients
/// @param intrinsics
/// Camera intrinsic parameters
/// @param distortion
/// Distortion coefficients
/// @param src
/// Input image data pointer
/// @param srcStride
/// Stride of a row in the source image (in bytes)
/// @param dst
/// Output image data pointer
/// @param dstStride
/// Stride of a row in the destination image (in bytes)
/// @param imageWidth
/// Image width (in pixels)
/// @param imageHeight
/// Image height (in pixels)
/// @return
/// A Status object that indicates whether the operation was successful
RETINIFY_API auto Undistort(const PinholeIntrinsics &intrinsics, const DistortionCoefficients &distortion, const std::uint8_t *src, std::size_t srcStride, std::uint8_t *dst, std::size_t dstStride, std::uint32_t imageWidth, std::uint32_t imageHeight) noexcept -> Status;

/// @brief
/// Perform stereo rectification for a pair of cameras
/// @param intrinsics1
/// First camera intrinsics
/// @param distortion1
/// First camera distortion
/// @param intrinsics2
/// Second camera intrinsics
/// @param distortion2
/// Second camera distortion
/// @param rotation
/// Rotation from the first to the second camera
/// @param translation
/// Translation from the first to the second camera
/// @param imageWidth
/// Image width (in pixels)
/// @param imageHeight
/// Image height (in pixels)
/// @param rotation1
/// Output rectification rotation for the first camera
/// @param rotation2
/// Output rectification rotation for the second camera
/// @param projectionMatrix1
/// Output projection matrix for the first camera
/// @param projectionMatrix2
/// Output projection matrix for the second camera
/// @param reprojectionMatrix
/// Output reprojection matrix
/// @param alpha
/// A free scaling parameter that controls cropping after rectification:
/// 0 keeps only valid pixels (no black borders),
/// 1 preserves the full original image (black borders included),
/// values between 0 and 1 yield intermediate results,
/// and -1 applies the default behavior
/// @return
/// A Status object that indicates whether the operation was successful
RETINIFY_API auto StereoRectify(const PinholeIntrinsics &intrinsics1, const DistortionCoefficients &distortion1, const PinholeIntrinsics &intrinsics2, const DistortionCoefficients &distortion2, const Mat3x3d &rotation, const Vec3d &translation, std::uint32_t imageWidth, std::uint32_t imageHeight, Mat3x3d &rotation1, Mat3x3d &rotation2, Mat3x4d &projectionMatrix1, Mat3x4d &projectionMatrix2, Mat4x4d &reprojectionMatrix, double alpha) noexcept -> Status;

/// @brief
/// Initialize undistort and rectify maps for image remapping
/// @param intrinsics
/// Camera intrinsics
/// @param distortion
/// Distortion coefficients
/// @param rotation
/// Rectification rotation
/// @param projectionMatrix
/// Projection matrix
/// @param mapX
/// Output map for x-coordinates
/// @param mapXStride
/// Stride of a row in mapX (in bytes)
/// @param mapY
/// Output map for y-coordinates
/// @param mapYStride
/// Stride of a row in mapY (in bytes)
/// @param imageWidth
/// Image width (in pixels)
/// @param imageHeight
/// Image height (in pixels)
/// @return
/// A Status object that indicates whether the operation was successful
RETINIFY_API auto InitUndistortRectifyMap(const PinholeIntrinsics &intrinsics, const DistortionCoefficients &distortion, const Mat3x3d &rotation, const Mat3x4d &projectionMatrix, float *mapX, std::size_t mapXStride, float *mapY, std::size_t mapYStride, std::uint32_t imageWidth, std::uint32_t imageHeight) noexcept -> Status;

/// @brief
/// Initialize identity maps for undistortion/rectification
/// @param mapX
/// Output map for x-coordinates
/// @param mapXStride
/// Stride of a row in mapX (in bytes)
/// @param mapY
/// Output map for y-coordinates
/// @param mapYStride
/// Stride of a row in mapY (in bytes)
/// @param imageWidth
/// Image width (in pixels)
/// @param imageHeight
/// Image height (in pixels)
/// @return
/// A Status object that indicates whether the operation was successful
RETINIFY_API auto InitIdentityMap(float *mapX, std::size_t mapXStride, float *mapY, std::size_t mapYStride, std::size_t imageWidth, std::size_t imageHeight) noexcept -> Status;
} // namespace retinify
