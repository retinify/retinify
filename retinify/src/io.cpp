// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/io.hpp"
#include "retinify/logging.hpp"
#include "retinify/nothrow.hpp"

#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

namespace retinify
{
namespace
{
constexpr int kJsonIndent = 2;

constexpr const char kKeyLeftIntrinsics[] = "left_intrinsics";
constexpr const char kKeyLeftDistortion[] = "left_distortion";
constexpr const char kKeyRightIntrinsics[] = "right_intrinsics";
constexpr const char kKeyRightDistortion[] = "right_distortion";
constexpr const char kKeyRotation[] = "rotation";
constexpr const char kKeyTranslation[] = "translation";
constexpr const char kKeyImageWidth[] = "image_width";
constexpr const char kKeyImageHeight[] = "image_height";
constexpr const char kKeyCalibrationError[] = "calibration_error";
constexpr const char kKeyLegacyReprojectionError[] = "reprojection_error";
constexpr const char kKeyCalibrationTime[] = "calibration_time";

constexpr const char kKeyFx[] = "fx";
constexpr const char kKeyFy[] = "fy";
constexpr const char kKeyCx[] = "cx";
constexpr const char kKeyCy[] = "cy";
constexpr const char kKeySkew[] = "skew";

constexpr const char kKeyK1[] = "k1";
constexpr const char kKeyK2[] = "k2";
constexpr const char kKeyP1[] = "p1";
constexpr const char kKeyP2[] = "p2";
constexpr const char kKeyK3[] = "k3";
constexpr const char kKeyK4[] = "k4";
constexpr const char kKeyK5[] = "k5";
constexpr const char kKeyK6[] = "k6";

[[nodiscard]] auto IsFilenameEmpty(const char *filename) noexcept -> bool
{
    return filename == nullptr || filename[0] == '\0';
}

[[nodiscard]] auto EnsureParentDirectory(const std::filesystem::path &filePath) -> Status
{
    const auto parent = filePath.parent_path();
    if (parent.empty())
    {
        return Status();
    }

    std::error_code fsError;
    std::filesystem::create_directories(parent, fsError);
    if (fsError)
    {
        return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
    }

    return Status();
}

template <typename T, typename Validator> [[nodiscard]] auto ReadValue(const nlohmann::json &obj, const char *key, Validator validator, T &out) -> bool
{
    const auto it = obj.find(key);
    if (it == obj.end() || !validator(*it))
    {
        return false;
    }

    out = it->get<T>();
    return true;
}

template <typename T> [[nodiscard]] auto ReadNumber(const nlohmann::json &obj, const char *key, T &out) -> bool
{
    return ReadValue(obj, key, [](const nlohmann::json &value) { return value.is_number(); }, out);
}

template <typename T> [[nodiscard]] auto ReadIntegral(const nlohmann::json &obj, const char *key, T &out) -> bool
{
    return ReadValue(obj, key, [](const nlohmann::json &value) { return value.is_number_integer() || value.is_number_unsigned(); }, out);
}

[[nodiscard]] auto SerializeIntrinsics(const PinholeIntrinsics &intrinsics) -> nlohmann::json
{
    return {{kKeyFx, intrinsics.fx}, //
            {kKeyFy, intrinsics.fy}, //
            {kKeyCx, intrinsics.cx}, //
            {kKeyCy, intrinsics.cy}, //
            {kKeySkew, intrinsics.skew}};
}

[[nodiscard]] auto SerializeDistortion(const DistortionCoefficients &distortion) -> nlohmann::json
{
    return {{kKeyK1, distortion.k1}, //
            {kKeyK2, distortion.k2}, //
            {kKeyP1, distortion.p1}, //
            {kKeyP2, distortion.p2}, //
            {kKeyK3, distortion.k3}, //
            {kKeyK4, distortion.k4}, //
            {kKeyK5, distortion.k5}, //
            {kKeyK6, distortion.k6}};
}

[[nodiscard]] auto SerializeCalibration(const CalibrationParameters &parameters) -> nlohmann::json
{
    return {{kKeyLeftIntrinsics, SerializeIntrinsics(parameters.leftIntrinsics)},   //
            {kKeyLeftDistortion, SerializeDistortion(parameters.leftDistortion)},   //
            {kKeyRightIntrinsics, SerializeIntrinsics(parameters.rightIntrinsics)}, //
            {kKeyRightDistortion, SerializeDistortion(parameters.rightDistortion)}, //
            {kKeyRotation, parameters.rotation},                                    //
            {kKeyTranslation, parameters.translation},                              //
            {kKeyImageWidth, parameters.imageWidth},                                //
            {kKeyImageHeight, parameters.imageHeight},                              //
            {kKeyCalibrationError, parameters.calibrationError},                    //
            {kKeyCalibrationTime, parameters.calibrationTime}};
}

[[nodiscard]] auto DeserializeIntrinsics(const nlohmann::json &value, PinholeIntrinsics &intrinsics) -> bool
{
    if (!value.is_object())
    {
        return false;
    }

    return ReadNumber(value, kKeyFx, intrinsics.fx) && //
           ReadNumber(value, kKeyFy, intrinsics.fy) && //
           ReadNumber(value, kKeyCx, intrinsics.cx) && //
           ReadNumber(value, kKeyCy, intrinsics.cy) && //
           ReadNumber(value, kKeySkew, intrinsics.skew);
}

[[nodiscard]] auto DeserializeDistortion(const nlohmann::json &value, DistortionCoefficients &distortion) -> bool
{
    if (!value.is_object())
    {
        return false;
    }

    return ReadNumber(value, kKeyK1, distortion.k1) && //
           ReadNumber(value, kKeyK2, distortion.k2) && //
           ReadNumber(value, kKeyP1, distortion.p1) && //
           ReadNumber(value, kKeyP2, distortion.p2) && //
           ReadNumber(value, kKeyK3, distortion.k3) && //
           ReadNumber(value, kKeyK4, distortion.k4) && //
           ReadNumber(value, kKeyK5, distortion.k5) && //
           ReadNumber(value, kKeyK6, distortion.k6);
}

template <std::size_t N> [[nodiscard]] auto DeserializeVector(const nlohmann::json &value, std::array<double, N> &out) -> bool
{
    if (!value.is_array() || value.size() != N)
    {
        return false;
    }

    for (std::size_t i = 0; i < N; ++i)
    {
        if (!value[i].is_number())
        {
            return false;
        }
        out[i] = value[i].get<double>();
    }

    return true;
}

template <std::size_t Rows, std::size_t Cols> [[nodiscard]] auto DeserializeMatrix(const nlohmann::json &value, std::array<std::array<double, Cols>, Rows> &out) -> bool
{
    if (!value.is_array() || value.size() != Rows)
    {
        return false;
    }

    for (std::size_t r = 0; r < Rows; ++r)
    {
        const auto &row = value[r];
        if (!row.is_array() || row.size() != Cols)
        {
            return false;
        }

        for (std::size_t c = 0; c < Cols; ++c)
        {
            if (!row[c].is_number())
            {
                return false;
            }
            out[r][c] = row[c].get<double>();
        }
    }

    return true;
}

[[nodiscard]] auto DeserializeCalibration(const nlohmann::json &doc, CalibrationParameters &parameters) -> bool
{
    if (!doc.is_object())
    {
        return false;
    }

    CalibrationParameters parsed{};

    const auto leftIntrinsicsIt = doc.find(kKeyLeftIntrinsics);
    if (leftIntrinsicsIt == doc.end() || !DeserializeIntrinsics(*leftIntrinsicsIt, parsed.leftIntrinsics))
    {
        return false;
    }

    const auto rightIntrinsicsIt = doc.find(kKeyRightIntrinsics);
    if (rightIntrinsicsIt == doc.end() || !DeserializeIntrinsics(*rightIntrinsicsIt, parsed.rightIntrinsics))
    {
        return false;
    }

    const auto leftDistortionIt = doc.find(kKeyLeftDistortion);
    if (leftDistortionIt == doc.end() || !DeserializeDistortion(*leftDistortionIt, parsed.leftDistortion))
    {
        return false;
    }

    const auto rightDistortionIt = doc.find(kKeyRightDistortion);
    if (rightDistortionIt == doc.end() || !DeserializeDistortion(*rightDistortionIt, parsed.rightDistortion))
    {
        return false;
    }

    const auto rotationIt = doc.find(kKeyRotation);
    if (rotationIt == doc.end() || !DeserializeMatrix(*rotationIt, parsed.rotation))
    {
        return false;
    }

    const auto translationIt = doc.find(kKeyTranslation);
    if (translationIt == doc.end() || !DeserializeVector(*translationIt, parsed.translation))
    {
        return false;
    }

    if (!ReadIntegral(doc, kKeyImageWidth, parsed.imageWidth))
    {
        return false;
    }

    if (!ReadIntegral(doc, kKeyImageHeight, parsed.imageHeight))
    {
        return false;
    }

    if (!ReadNumber(doc, kKeyCalibrationError, parsed.calibrationError) && !ReadNumber(doc, kKeyLegacyReprojectionError, parsed.calibrationError))
    {
        return false;
    }

    if (!ReadIntegral(doc, kKeyCalibrationTime, parsed.calibrationTime))
    {
        return false;
    }

    parameters = parsed;
    return true;
}
} // namespace

auto SaveCalibrationParameters(const char *filename, const CalibrationParameters &parameters) noexcept -> Status
{
    return NoThrow([&]() -> Status {
        if (IsFilenameEmpty(filename))
        {
            return Status{StatusCategory::USER, StatusCode::INVALID_ARGUMENT};
        }

        const std::filesystem::path targetPath(filename);

        const auto dirStatus = EnsureParentDirectory(targetPath);
        if (!dirStatus.IsOK())
        {
            return dirStatus;
        }

        std::ofstream out{targetPath, std::ios::trunc};
        if (!out.is_open())
        {
            return Status{StatusCategory::SYSTEM, StatusCode::FAIL};
        }

        const auto json = SerializeCalibration(parameters);
        out << json.dump(kJsonIndent) << '\n';

        out.flush();
        if (!out.good())
        {
            return Status{StatusCategory::SYSTEM, StatusCode::FAIL};
        }

        return Status{};
    });
}

auto LoadCalibrationParameters(const char *filename, CalibrationParameters &parameters) noexcept -> Status
{
    return NoThrow([&]() -> Status {
        if (IsFilenameEmpty(filename))
        {
            return Status{StatusCategory::USER, StatusCode::INVALID_ARGUMENT};
        }

        std::ifstream in(filename, std::ios::in);
        if (!in.is_open())
        {
            return Status{StatusCategory::SYSTEM, StatusCode::FAIL};
        }

        const auto doc = nlohmann::json::parse(in, nullptr, false);
        if (doc.is_discarded())
        {
            return Status{StatusCategory::SYSTEM, StatusCode::FAIL};
        }

        if (!DeserializeCalibration(doc, parameters))
        {
            return Status{StatusCategory::SYSTEM, StatusCode::FAIL};
        }

        return Status{};
    });
}
} // namespace retinify
