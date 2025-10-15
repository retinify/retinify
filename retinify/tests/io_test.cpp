// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-eula

#include "retinify/io.hpp"

#include <algorithm>
#include <bit>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace retinify
{
namespace
{
class ScopedTempDir
{
  public:
    ScopedTempDir()
    {
        const auto base = std::filesystem::temp_directory_path();
        std::random_device rd;

        for (int attempt = 0; attempt < 32; ++attempt)
        {
            const auto candidate = base / ("retinify-io-test-" + std::to_string(rd()) + "-" + std::to_string(attempt));

            std::error_code ec;
            if (std::filesystem::create_directory(candidate, ec))
            {
                path_ = candidate;
                return;
            }

            if (!ec && std::filesystem::is_directory(candidate))
            {
                path_ = candidate;
                return;
            }
        }

        throw std::runtime_error("Unable to create temporary directory for io tests");
    }

    ~ScopedTempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    [[nodiscard]] auto path() const noexcept -> const std::filesystem::path &
    {
        return path_;
    }

  private:
    std::filesystem::path path_;
};

[[nodiscard]] auto MakeSampleParameters() -> CalibrationParameters
{
    CalibrationParameters params{};

    params.leftIntrinsics.fx = 610.25;
    params.leftIntrinsics.fy = 612.5;
    params.leftIntrinsics.cx = 320.75;
    params.leftIntrinsics.cy = 240.125;
    params.leftIntrinsics.skew = -0.0125;

    params.leftDistortion.k1 = -0.01;
    params.leftDistortion.k2 = 0.002;
    params.leftDistortion.p1 = -0.0003;
    params.leftDistortion.p2 = 0.0004;
    params.leftDistortion.k3 = -0.0005;
    params.leftDistortion.k4 = 0.0006;
    params.leftDistortion.k5 = -0.0007;
    params.leftDistortion.k6 = 0.0008;

    params.rightIntrinsics.fx = 605.75;
    params.rightIntrinsics.fy = 607.0;
    params.rightIntrinsics.cx = 318.25;
    params.rightIntrinsics.cy = 242.5;
    params.rightIntrinsics.skew = 0.025;

    params.rightDistortion.k1 = 0.011;
    params.rightDistortion.k2 = -0.0022;
    params.rightDistortion.p1 = 0.00033;
    params.rightDistortion.p2 = -0.00044;
    params.rightDistortion.k3 = 0.00055;
    params.rightDistortion.k4 = -0.00066;
    params.rightDistortion.k5 = 0.00077;
    params.rightDistortion.k6 = -0.00088;

    params.rotation = {{{0.9998, -0.0175, 0.0042}, {0.0176, 0.9997, -0.0123}, {-0.0040, 0.0124, 0.9999}}};

    params.translation = {0.105, -0.208, 0.315};

    params.imageWidth = 640;
    params.imageHeight = 480;
    params.reprojectionError = 0.25;
    params.calibrationTime = 1720000000123456789ull;
    const char left[] = "LEFT1234567890";
    const char right[] = "RIGHT0987654321";
    std::copy(std::begin(left), std::end(left), params.leftCameraSerial.begin());
    std::copy(std::begin(right), std::end(right), params.rightCameraSerial.begin());

    return params;
}

auto ExpectIntrinsicsEqual(const Intrinsics &expected, const Intrinsics &actual) -> void
{
    EXPECT_DOUBLE_EQ(expected.fx, actual.fx);
    EXPECT_DOUBLE_EQ(expected.fy, actual.fy);
    EXPECT_DOUBLE_EQ(expected.cx, actual.cx);
    EXPECT_DOUBLE_EQ(expected.cy, actual.cy);
    EXPECT_DOUBLE_EQ(expected.skew, actual.skew);
}

auto ExpectDistortionEqual(const Distortion &expected, const Distortion &actual) -> void
{
    EXPECT_DOUBLE_EQ(expected.k1, actual.k1);
    EXPECT_DOUBLE_EQ(expected.k2, actual.k2);
    EXPECT_DOUBLE_EQ(expected.p1, actual.p1);
    EXPECT_DOUBLE_EQ(expected.p2, actual.p2);
    EXPECT_DOUBLE_EQ(expected.k3, actual.k3);
    EXPECT_DOUBLE_EQ(expected.k4, actual.k4);
    EXPECT_DOUBLE_EQ(expected.k5, actual.k5);
    EXPECT_DOUBLE_EQ(expected.k6, actual.k6);
}

auto ExpectRotationEqual(const Mat3x3d &expected, const Mat3x3d &actual) -> void
{
    for (std::size_t row = 0; row < expected.size(); ++row)
    {
        for (std::size_t col = 0; col < expected[row].size(); ++col)
        {
            EXPECT_DOUBLE_EQ(expected[row][col], actual[row][col]) << "rotation mismatch at (" << row << ", " << col << ")";
        }
    }
}

auto ExpectTranslationEqual(const Vec3d &expected, const Vec3d &actual) -> void
{
    for (std::size_t idx = 0; idx < expected.size(); ++idx)
    {
        EXPECT_DOUBLE_EQ(expected[idx], actual[idx]) << "translation mismatch at index " << idx;
    }
}

auto ExpectParametersEqual(const CalibrationParameters &expected, const CalibrationParameters &actual) -> void
{
    ExpectIntrinsicsEqual(expected.leftIntrinsics, actual.leftIntrinsics);
    ExpectDistortionEqual(expected.leftDistortion, actual.leftDistortion);
    ExpectIntrinsicsEqual(expected.rightIntrinsics, actual.rightIntrinsics);
    ExpectDistortionEqual(expected.rightDistortion, actual.rightDistortion);
    ExpectRotationEqual(expected.rotation, actual.rotation);
    ExpectTranslationEqual(expected.translation, actual.translation);
    EXPECT_EQ(expected.imageWidth, actual.imageWidth);
    EXPECT_EQ(expected.imageHeight, actual.imageHeight);
    EXPECT_DOUBLE_EQ(expected.reprojectionError, actual.reprojectionError);
    EXPECT_EQ(expected.calibrationTime, actual.calibrationTime);
    EXPECT_EQ(std::string(expected.leftCameraSerial.data(), expected.leftCameraSerial.size()), std::string(actual.leftCameraSerial.data(), actual.leftCameraSerial.size()));
    EXPECT_EQ(std::string(expected.rightCameraSerial.data(), expected.rightCameraSerial.size()), std::string(actual.rightCameraSerial.data(), actual.rightCameraSerial.size()));
}

auto Byteswap32(std::uint32_t value) -> std::uint32_t
{
    return ((value & 0x000000FFu) << 24) | ((value & 0x0000FF00u) << 8) | ((value & 0x00FF0000u) >> 8) | ((value & 0xFF000000u) >> 24);
}
} // namespace

TEST(IoTest, SaveAndLoadRoundTrip)
{
    ScopedTempDir tempDir;
    const auto filePath = tempDir.path() / "calibration.bin";
    const auto params = MakeSampleParameters();

    const auto saveStatus = SaveCalibrationParameters(filePath.string().c_str(), params);
    ASSERT_TRUE(saveStatus.IsOK());

    CalibrationParameters loaded{};
    const auto loadStatus = LoadCalibrationParameters(filePath.string().c_str(), loaded);
    ASSERT_TRUE(loadStatus.IsOK());

    ExpectParametersEqual(params, loaded);
}

TEST(IoTest, SaveRejectsInvalidFilename)
{
    const auto params = MakeSampleParameters();

    const auto nullStatus = SaveCalibrationParameters(nullptr, params);
    EXPECT_FALSE(nullStatus.IsOK());
    EXPECT_EQ(nullStatus.Category(), StatusCategory::USER);
    EXPECT_EQ(nullStatus.Code(), StatusCode::INVALID_ARGUMENT);

    const auto emptyStatus = SaveCalibrationParameters("", params);
    EXPECT_FALSE(emptyStatus.IsOK());
    EXPECT_EQ(emptyStatus.Category(), StatusCategory::USER);
    EXPECT_EQ(emptyStatus.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST(IoTest, LoadRejectsInvalidFilename)
{
    CalibrationParameters params{};

    const auto nullStatus = LoadCalibrationParameters(nullptr, params);
    EXPECT_FALSE(nullStatus.IsOK());
    EXPECT_EQ(nullStatus.Category(), StatusCategory::USER);
    EXPECT_EQ(nullStatus.Code(), StatusCode::INVALID_ARGUMENT);

    const auto emptyStatus = LoadCalibrationParameters("", params);
    EXPECT_FALSE(emptyStatus.IsOK());
    EXPECT_EQ(emptyStatus.Category(), StatusCategory::USER);
    EXPECT_EQ(emptyStatus.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST(IoTest, SaveFailsWhenParentIsFile)
{
    ScopedTempDir tempDir;
    const auto parent = tempDir.path() / "not-a-directory";
    {
        std::ofstream marker(parent);
        ASSERT_TRUE(marker.is_open());
    }

    const auto target = parent / "calibration.bin";
    const auto params = MakeSampleParameters();

    const auto status = SaveCalibrationParameters(target.string().c_str(), params);
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::SYSTEM);
    EXPECT_EQ(status.Code(), StatusCode::FAIL);

    std::error_code ec;
    EXPECT_FALSE(std::filesystem::exists(target, ec));
}

TEST(IoTest, LoadFailsForMissingFile)
{
    ScopedTempDir tempDir;
    const auto missing = tempDir.path() / "does-not-exist.cal";

    CalibrationParameters params{};
    const auto status = LoadCalibrationParameters(missing.string().c_str(), params);

    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::SYSTEM);
    EXPECT_EQ(status.Code(), StatusCode::FAIL);
}

TEST(IoTest, LoadFailsForInvalidMagic)
{
    ScopedTempDir tempDir;
    const auto filePath = tempDir.path() / "corrupt-magic.cal";
    const auto params = MakeSampleParameters();
    ASSERT_TRUE(SaveCalibrationParameters(filePath.string().c_str(), params).IsOK());

    std::fstream file(filePath, std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(file.is_open());
    file.seekp(static_cast<std::streamoff>(0));
    file.put('X');
    file.close();

    CalibrationParameters parsed{};
    const auto status = LoadCalibrationParameters(filePath.string().c_str(), parsed);

    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::SYSTEM);
    EXPECT_EQ(status.Code(), StatusCode::FAIL);
}

TEST(IoTest, LoadFailsForUnsupportedVersion)
{
    ScopedTempDir tempDir;
    const auto filePath = tempDir.path() / "bad-version.cal";
    const auto params = MakeSampleParameters();
    ASSERT_TRUE(SaveCalibrationParameters(filePath.string().c_str(), params).IsOK());

    std::fstream file(filePath, std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(file.is_open());
    file.seekp(static_cast<std::streamoff>(13));

    std::uint32_t newVersion = 999;
    if constexpr (std::endian::native != std::endian::little)
    {
        newVersion = Byteswap32(newVersion);
    }
    file.write(reinterpret_cast<const char *>(&newVersion), sizeof(newVersion));
    file.close();

    CalibrationParameters parsed{};
    const auto status = LoadCalibrationParameters(filePath.string().c_str(), parsed);

    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::SYSTEM);
    EXPECT_EQ(status.Code(), StatusCode::FAIL);
}
} // namespace retinify
