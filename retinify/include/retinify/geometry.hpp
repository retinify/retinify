// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-EULA

#pragma once

#include "attributes.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace retinify
{
/// @brief
/// 2D vector (double).
using Vec2d = std::array<double, 2>;

/// @brief
/// 3D vector (double).
using Vec3d = std::array<double, 3>;

/// @brief
/// 2D point (double).
using Point2d = std::array<double, 2>;

/// @brief
/// 3D point (double).
using Point3d = std::array<double, 3>;

/// @brief
/// 3x3 matrix (double, row-major).
using Mat3x3d = std::array<std::array<double, 3>, 3>;

/// @brief
/// 3x4 matrix (double, row-major).
using Mat3x4d = std::array<std::array<double, 4>, 3>;

/// @brief
/// 4x4 matrix (double, row-major).
using Mat4x4d = std::array<std::array<double, 4>, 4>;

/// @brief
/// 2D vector (float).
using Vec2f = std::array<float, 2>;

/// @brief
/// 3D vector (float).
using Vec3f = std::array<float, 3>;

/// @brief
/// 2D point (float).
using Point2f = std::array<float, 2>;

/// @brief
/// 3D point (float).
using Point3f = std::array<float, 3>;

/// @brief
/// 3x3 matrix (float, row-major).
using Mat3x3f = std::array<std::array<float, 3>, 3>;

/// @brief
/// 3x4 matrix (float, row-major).
using Mat3x4f = std::array<std::array<float, 4>, 3>;

/// @brief
/// 4x4 matrix (float, row-major).
using Mat4x4f = std::array<std::array<float, 4>, 4>;

/// @brief
/// Rectangle structure.
/// @tparam T
/// Type of the rectangle coordinates and dimensions.
template <typename T> struct Rect2
{
    /// @brief X coordinate of the top-left corner.
    T x{0};
    /// @brief Y coordinate of the top-left corner.
    T y{0};
    /// @brief Width of the rectangle.
    T width{0};
    /// @brief Height of the rectangle.
    T height{0};
};

/// @brief
/// 2D rectangle (int).
using Rect2i = Rect2<std::int32_t>;

/// @brief
/// 2D rectangle (double).
using Rect2d = Rect2<double>;

/// @brief
/// Create a 3x3 identity matrix.
/// @return
/// 3x3 identity matrix.
RETINIFY_API auto Identity() noexcept -> Mat3x3d;

/// @brief
/// Compute the determinant of a 3x3 matrix.
/// @param mat
/// 3x3 matrix.
/// @return
/// Determinant value.
RETINIFY_API auto Determinant(const Mat3x3d &mat) noexcept -> double;

/// @brief
/// Transpose a 3x3 matrix.
/// @param mat
/// 3x3 matrix.
/// @return
/// Transposed 3x3 matrix.
RETINIFY_API auto Transpose(const Mat3x3d &mat) noexcept -> Mat3x3d;

/// @brief
/// Multiply two 3x3 matrices.
/// @param mat1
/// First 3x3 matrix.
/// @param mat2
/// Second 3x3 matrix.
/// @return
/// 3x3 matrix.
RETINIFY_API auto Multiply(const Mat3x3d &mat1, const Mat3x3d &mat2) noexcept -> Mat3x3d;

/// @brief
/// Multiply a 3x3 matrix and a 3D vector.
/// @param mat
/// 3x3 matrix.
/// @param vec
/// 3D vector.
/// @return
/// 3D vector.
RETINIFY_API auto Multiply(const Mat3x3d &mat, const Vec3d &vec) noexcept -> Vec3d;

/// @brief
/// Scale a 3D vector by a scalar value.
/// @param vec
/// 3D vector
/// @param scale
/// Scalar value
/// @return
/// Scaled 3D vector.
RETINIFY_API auto Scale(const Vec3d &vec, double scale) noexcept -> Vec3d;

/// @brief
/// Compute the length (magnitude) of a 3D vector.
/// @param vec
/// 3D vector.
/// @return
/// Length value.
RETINIFY_API auto Length(const Vec3d &vec) noexcept -> double;

/// @brief
/// Normalize a 3D vector to unit length.
/// @param vec
/// 3D vector.
/// @return
/// Normalized 3D vector.
RETINIFY_API auto Normalize(const Vec3d &vec) noexcept -> Vec3d;

/// @brief
/// Compute the dot product of two 3D vectors.
/// @param vec1
/// First 3D vector.
/// @param vec2
/// Second 3D vector.
/// @return
/// Dot product value.
RETINIFY_API auto Dot(const Vec3d &vec1, const Vec3d &vec2) noexcept -> double;

/// @brief
/// Compute the cross product of two 3D vectors.
/// @param vec1
/// First 3D vector.
/// @param vec2
/// Second 3D vector.
/// @return
/// Cross product vector.
RETINIFY_API auto Cross(const Vec3d &vec1, const Vec3d &vec2) noexcept -> Vec3d;

/// @brief
/// Create a 3x3 skew-symmetric matrix from a 3D rotation vector.
/// @param omega
/// 3D rotation vector.
/// @return
/// 3x3 skew-symmetric matrix.
RETINIFY_API auto Hat(const Vec3d &omega) noexcept -> Mat3x3d;

/// @brief
/// Convert a 3x3 skew-symmetric matrix to a 3D rotation vector.
/// @param skew
/// 3x3 skew-symmetric matrix.
/// @return
/// 3D rotation vector.
RETINIFY_API auto Vee(const Mat3x3d &skew) noexcept -> Vec3d;

/// @brief
/// Compute the matrix exponential of a 3D rotation vector.
/// @param omega
/// 3D rotation vector.
/// @return
/// 3x3 rotation matrix.
RETINIFY_API auto Exp(const Vec3d &omega) noexcept -> Mat3x3d;

/// @brief
/// Compute the matrix logarithm of a 3x3 rotation matrix.
/// @param rotation
/// 3x3 rotation matrix.
/// @return
/// 3D rotation vector.
RETINIFY_API auto Log(const Mat3x3d &rotation) noexcept -> Vec3d;

/// @brief
/// Camera intrinsic parameters with focal lengths, principal point, and skew.
struct Intrinsics
{
    /// @brief
    /// Focal length in x [pixels]
    double fx{0};
    /// @brief
    /// Focal length in y [pixels]
    double fy{0};
    /// @brief
    /// Principal point x-coordinate [pixels]
    double cx{0};
    /// @brief
    /// Principal point y-coordinate [pixels]
    double cy{0};
    /// @brief
    /// Skew coefficient
    double skew{0};

    [[nodiscard]] auto operator==(const Intrinsics &other) const noexcept -> bool
    {
        return fx == other.fx && //
               fy == other.fy && //
               cx == other.cx && //
               cy == other.cy && //
               skew == other.skew;
    }
};

/// @brief
/// Rational distortion model with 8 coefficients: (k1, k2, p1, p2, k3, k4, k5, k6).
struct Distortion
{
    double k1{0};
    double k2{0};
    double p1{0};
    double p2{0};
    double k3{0};
    double k4{0};
    double k5{0};
    double k6{0};

    [[nodiscard]] auto operator==(const Distortion &other) const noexcept -> bool
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
/// Fisheye distortion model with 4 coefficients (k1, k2, k3, k4).
struct DistortionFisheye
{
    double k1{0};
    double k2{0};
    double k3{0};
    double k4{0};
};

/// @brief
/// Stereo camera calibration parameters.
struct CalibrationParameters
{
    /// @brief
    /// Intrinsics for the left camera
    Intrinsics leftIntrinsics;
    /// @brief
    /// Distortion for the left camera
    Distortion leftDistortion;
    /// @brief
    /// Intrinsics for the right camera
    Intrinsics rightIntrinsics;
    /// @brief
    /// Distortion for the right camera
    Distortion rightDistortion;
    /// @brief
    /// Rotation from the left to the right camera
    Mat3x3d rotation;
    /// @brief
    /// Translation from the left to the right camera
    Vec3d translation;
    /// @brief
    /// Image width [pixels]
    std::uint32_t imageWidth{};
    /// @brief
    /// Image height [pixels]
    std::uint32_t imageHeight{};
    /// @brief
    /// Root mean square reprojection error [pixels]
    double reprojectionError{};
    /// @brief
    /// Calibration timestamp in Unix time [nanoseconds]
    std::uint64_t calibrationTime{};
    /// @brief
    /// Left camera hardware serial
    std::array<char, 128> leftCameraSerial{};
    /// @brief
    /// Right camera hardware serial
    std::array<char, 128> rightCameraSerial{};

    [[nodiscard]] auto operator==(const CalibrationParameters &other) const noexcept -> bool
    {
        return leftIntrinsics == other.leftIntrinsics &&       //
               leftDistortion == other.leftDistortion &&       //
               rightIntrinsics == other.rightIntrinsics &&     //
               rightDistortion == other.rightDistortion &&     //
               rotation == other.rotation &&                   //
               translation == other.translation &&             //
               imageWidth == other.imageWidth &&               //
               imageHeight == other.imageHeight &&             //
               reprojectionError == other.reprojectionError && //
               calibrationTime == other.calibrationTime &&     //
               leftCameraSerial == other.leftCameraSerial &&   //
               rightCameraSerial == other.rightCameraSerial;   //
    }
};

/// @brief
/// Undistort a 2D point using the given camera intrinsics and distortion parameters.
/// @param intrinsics
/// Camera intrinsic parameters.
/// @param distortion
/// Distortion parameters.
/// @param pixel
/// Distorted 2D point in pixel coordinates.
/// @return
/// Undistorted 2D point in pixel coordinates.
RETINIFY_API auto UndistortPoint(const Intrinsics &intrinsics, const Distortion &distortion, const Point2d &pixel) noexcept -> Point2d;

/// @brief
/// Perform stereo rectification for a pair of cameras.
/// @param intrinsics1
/// First camera intrinsics.
/// @param distortion1
/// First camera distortion.
/// @param intrinsics2
/// Second camera intrinsics.
/// @param distortion2
/// Second camera distortion.
/// @param rotation
/// Rotation from the first to the second camera.
/// @param translation
/// Translation from the first to the second camera.
/// @param imageWidth
/// Image width in pixels.
/// @param imageHeight
/// Image height in pixels.
/// @param rotation1
/// Output rectification rotation for the first camera.
/// @param rotation2
/// Output rectification rotation for the second camera.
/// @param projectionMatrix1
/// Output projection matrix for the first camera.
/// @param projectionMatrix2
/// Output projection matrix for the second camera.
/// @param mappingMatrix
/// Output mapping matrix.
/// @param alpha
/// A free scaling parameter that controls cropping after rectification:
/// 0 keeps only valid pixels (no black borders),
/// 1 preserves the full original image (black borders included),
/// values between 0 and 1 yield intermediate results,
/// and -1 applies the default behavior.
RETINIFY_API auto StereoRectify(const Intrinsics &intrinsics1, const Distortion &distortion1, //
                                const Intrinsics &intrinsics2, const Distortion &distortion2, //
                                const Mat3x3d &rotation, const Vec3d &translation,            //
                                std::uint32_t imageWidth, std::uint32_t imageHeight,          //
                                Mat3x3d &rotation1, Mat3x3d &rotation2,                       //
                                Mat3x4d &projectionMatrix1, Mat3x4d &projectionMatrix2,       //
                                Mat4x4d &mappingMatrix, double alpha) noexcept -> void;

/// @brief
/// Initialize undistort and rectify maps for image remapping.
/// @param intrinsics
/// Camera intrinsics
/// @param distortion
/// Distortion parameters
/// @param rotation
/// Rectification rotation
/// @param projectionMatrix
/// Projection matrix
/// @param imageWidth
/// Image width in pixels.
/// @param imageHeight
/// Image height in pixels.
/// @param mapx
/// Output map for x-coordinates
/// @param mapxStride
/// Stride (in bytes) of a row in mapx
/// @param mapy
/// Output map for y-coordinates
/// @param mapyStride
/// Stride (in bytes) of a row in mapy
/// @return
RETINIFY_API auto InitUndistortRectifyMap(const Intrinsics &intrinsics, const Distortion &distortion, //
                                          const Mat3x3d &rotation, const Mat3x4d &projectionMatrix,   //
                                          std::uint32_t imageWidth, std::uint32_t imageHeight,        //
                                          float *mapx, std::size_t mapxStride,                        //
                                          float *mapy, std::size_t mapyStride) noexcept -> void;

/// @brief
/// Initialize identity maps for undistortion/rectification.
/// @param mapx
/// Output map for x-coordinates
/// @param mapxStride
/// Stride (in bytes) of a row in mapx
/// @param mapy
/// Output map for y-coordinates
/// @param mapyStride
/// Stride (in bytes) of a row in mapy
/// @param imageWidth
/// Image width in pixels
/// @param imageHeight
/// Image height in pixels
RETINIFY_API auto InitIdentityMap(float *mapx, std::size_t mapxStride, //
                                  float *mapy, std::size_t mapyStride, //
                                  std::size_t imageWidth, std::size_t imageHeight) noexcept -> void;
} // namespace retinify
