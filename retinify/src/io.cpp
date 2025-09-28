// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/io.hpp"
#include "retinify/logging.hpp"

#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>

namespace retinify
{
namespace
{
constexpr std::uint32_t kFileMagicElementCount = 13;
constexpr std::array<std::uint8_t, kFileMagicElementCount> kFileMagic{
    static_cast<std::uint8_t>('R'), //
    static_cast<std::uint8_t>('E'), //
    static_cast<std::uint8_t>('T'), //
    static_cast<std::uint8_t>('I'), //
    static_cast<std::uint8_t>('N'), //
    static_cast<std::uint8_t>('I'), //
    static_cast<std::uint8_t>('F'), //
    static_cast<std::uint8_t>('Y'), //
    static_cast<std::uint8_t>('C'), //
    static_cast<std::uint8_t>('A'), //
    static_cast<std::uint8_t>('L'), //
    static_cast<std::uint8_t>('I'), //
    static_cast<std::uint8_t>('B'), //
};

constexpr std::uint32_t kFileFormatVersion = 1;

constexpr std::size_t kIdxLeftFx = 0;
constexpr std::size_t kIdxLeftFy = 1;
constexpr std::size_t kIdxLeftCx = 2;
constexpr std::size_t kIdxLeftCy = 3;
constexpr std::size_t kIdxLeftSkew = 4;

constexpr std::size_t kIdxLeftK1 = 5;
constexpr std::size_t kIdxLeftK2 = 6;
constexpr std::size_t kIdxLeftP1 = 7;
constexpr std::size_t kIdxLeftP2 = 8;
constexpr std::size_t kIdxLeftK3 = 9;
constexpr std::size_t kIdxLeftK4 = 10;
constexpr std::size_t kIdxLeftK5 = 11;
constexpr std::size_t kIdxLeftK6 = 12;

constexpr std::size_t kIdxRightFx = 13;
constexpr std::size_t kIdxRightFy = 14;
constexpr std::size_t kIdxRightCx = 15;
constexpr std::size_t kIdxRightCy = 16;
constexpr std::size_t kIdxRightSkew = 17;

constexpr std::size_t kIdxRightK1 = 18;
constexpr std::size_t kIdxRightK2 = 19;
constexpr std::size_t kIdxRightP1 = 20;
constexpr std::size_t kIdxRightP2 = 21;
constexpr std::size_t kIdxRightK3 = 22;
constexpr std::size_t kIdxRightK4 = 23;
constexpr std::size_t kIdxRightK5 = 24;
constexpr std::size_t kIdxRightK6 = 25;

constexpr std::size_t kIdxRotation00 = 26;
constexpr std::size_t kIdxRotation01 = 27;
constexpr std::size_t kIdxRotation02 = 28;
constexpr std::size_t kIdxRotation10 = 29;
constexpr std::size_t kIdxRotation11 = 30;
constexpr std::size_t kIdxRotation12 = 31;
constexpr std::size_t kIdxRotation20 = 32;
constexpr std::size_t kIdxRotation21 = 33;
constexpr std::size_t kIdxRotation22 = 34;

constexpr std::size_t kIdxTranslationX = 35;
constexpr std::size_t kIdxTranslationY = 36;
constexpr std::size_t kIdxTranslationZ = 37;

constexpr std::size_t kStereoCameraParametersElementCount = 38;

constexpr std::size_t kCameraSerialSize = 128;

constexpr std::uintmax_t kExpectedFileSize = (kFileMagicElementCount * sizeof(std::uint8_t)) +        // file magic
                                             sizeof(std::uint32_t) +                                  // file format version
                                             (kStereoCameraParametersElementCount * sizeof(double)) + // stereo camera parameters
                                             (2 * sizeof(std::uint32_t)) +                            // imageWidth, imageHeight
                                             sizeof(double) +                                         // reprojectionError
                                             sizeof(std::uint64_t) +                                  // calibrationTime
                                             (2 * kCameraSerialSize * sizeof(char));                  // left & right serial

constexpr bool kIsNativeLittleEndian = (std::endian::native == std::endian::little);

// Create parent directories if needed
[[nodiscard]] auto CreateParentDirectories(const std::filesystem::path &filePath) -> Status
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

// Compose a unique temporary path candidate next to the target file.
[[nodiscard]] auto ComposeTemporaryCandidate(const std::filesystem::path &parent, std::string baseName, int attempt) -> std::filesystem::path
{
    if (baseName.empty())
    {
        baseName = "retinify-calibration";
    }

    std::string candidate = baseName + ".tmp";
    if (attempt > 0)
    {
        candidate += '-' + std::to_string(attempt);
    }

    return parent.empty() ? std::filesystem::path(candidate) : parent / candidate;
}

// Find an unused temporary path for the target file.
[[nodiscard]] auto MakeTemporaryPath(const std::filesystem::path &targetPath, std::error_code &ec) -> std::filesystem::path
{
    const auto parent = targetPath.parent_path();
    std::string baseName = targetPath.filename().string();

    constexpr int kMaxTemporaryPathAttempts = 64;
    for (int attempt = 0; attempt < kMaxTemporaryPathAttempts; ++attempt)
    {
        const auto candidate = ComposeTemporaryCandidate(parent, baseName, attempt);

        std::error_code existsError;
        const bool exists = std::filesystem::exists(candidate, existsError);
        if (existsError)
        {
            ec = existsError;
            return {};
        }

        if (!exists)
        {
            ec.clear();
            return candidate;
        }
    }

    ec = std::make_error_code(std::errc::file_exists);
    return {};
}

// Delete the provided path when present, ignoring errors.
auto RemovePathIfExists(const std::filesystem::path &path) noexcept -> void
{
    if (path.empty())
    {
        return;
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

[[nodiscard]] constexpr auto Byteswap32(std::uint32_t value) noexcept -> std::uint32_t
{
    return ((value & 0x000000FFu) << 24) | //
           ((value & 0x0000FF00u) << 8) |  //
           ((value & 0x00FF0000u) >> 8) |  //
           ((value & 0xFF000000u) >> 24);
}

[[nodiscard]] constexpr auto Byteswap64(std::uint64_t value) noexcept -> std::uint64_t
{
    return ((value & 0x00000000000000FFull) << 56) | //
           ((value & 0x000000000000FF00ull) << 40) | //
           ((value & 0x0000000000FF0000ull) << 24) | //
           ((value & 0x00000000FF000000ull) << 8) |  //
           ((value & 0x000000FF00000000ull) >> 8) |  //
           ((value & 0x0000FF0000000000ull) >> 24) | //
           ((value & 0x00FF000000000000ull) >> 40) | //
           ((value & 0xFF00000000000000ull) >> 56);
}

[[nodiscard]] auto WriteUint32(std::ostream &stream, std::uint32_t value) -> bool
{
    auto temp = value;
    if constexpr (!kIsNativeLittleEndian)
    {
        temp = Byteswap32(temp);
    }

    stream.write(reinterpret_cast<const char *>(&temp), sizeof(temp));
    return stream.good();
}

[[nodiscard]] auto WriteUint64(std::ostream &stream, std::uint64_t value) -> bool
{
    auto temp = value;
    if constexpr (!kIsNativeLittleEndian)
    {
        temp = Byteswap64(temp);
    }

    stream.write(reinterpret_cast<const char *>(&temp), sizeof(temp));
    return stream.good();
}

[[nodiscard]] auto WriteDouble(std::ostream &stream, double value) -> bool
{
    auto bits = std::bit_cast<std::uint64_t>(value);
    if constexpr (!kIsNativeLittleEndian)
    {
        bits = Byteswap64(bits);
    }

    stream.write(reinterpret_cast<const char *>(&bits), sizeof(bits));
    return stream.good();
}

[[nodiscard]] auto ReadUint32(std::istream &stream, std::uint32_t &value) -> bool
{
    std::uint32_t temp{};
    stream.read(reinterpret_cast<char *>(&temp), sizeof(temp));

    if (!stream)
    {
        return false;
    }

    if constexpr (!kIsNativeLittleEndian)
    {
        temp = Byteswap32(temp);
    }

    value = temp;
    return true;
}

[[nodiscard]] auto ReadUint64(std::istream &stream, std::uint64_t &value) -> bool
{
    std::uint64_t temp{};
    stream.read(reinterpret_cast<char *>(&temp), sizeof(temp));

    if (!stream)
    {
        return false;
    }

    if constexpr (!kIsNativeLittleEndian)
    {
        temp = Byteswap64(temp);
    }

    value = temp;
    return true;
}

[[nodiscard]] auto ReadDouble(std::istream &stream, double &value) -> bool
{
    std::uint64_t bits{};
    stream.read(reinterpret_cast<char *>(&bits), sizeof(bits));

    if (!stream)
    {
        return false;
    }

    if constexpr (!kIsNativeLittleEndian)
    {
        bits = Byteswap64(bits);
    }

    value = std::bit_cast<double>(bits);
    return true;
}

template <std::size_t N> [[nodiscard]] auto WriteDoubleArray(std::ostream &stream, const std::array<double, N> &values) -> bool
{
    for (const double value : values)
    {
        if (!WriteDouble(stream, value))
        {
            return false;
        }
    }

    return true;
}

template <std::size_t N> [[nodiscard]] auto ReadDoubleArray(std::istream &stream, std::array<double, N> &values) -> bool
{
    for (auto &value : values)
    {
        if (!ReadDouble(stream, value))
        {
            return false;
        }
    }

    return true;
}

template <std::size_t N> [[nodiscard]] auto WriteCharArray(std::ostream &stream, const std::array<char, N> &buffer) -> bool
{
    stream.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    return stream.good();
}

template <std::size_t N> [[nodiscard]] auto ReadCharArray(std::istream &stream, std::array<char, N> &buffer) -> bool
{
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    if (!stream)
    {
        return false;
    }
    return true;
}

[[nodiscard]] auto FlattenStereoCameraParameters(const CalibrationParameters &parameters) noexcept -> std::array<double, kStereoCameraParametersElementCount>
{
    std::array<double, kStereoCameraParametersElementCount> values{};
    values[kIdxLeftFx] = parameters.leftIntrinsics.fx;
    values[kIdxLeftFy] = parameters.leftIntrinsics.fy;
    values[kIdxLeftCx] = parameters.leftIntrinsics.cx;
    values[kIdxLeftCy] = parameters.leftIntrinsics.cy;
    values[kIdxLeftSkew] = parameters.leftIntrinsics.skew;

    values[kIdxLeftK1] = parameters.leftDistortion.k1;
    values[kIdxLeftK2] = parameters.leftDistortion.k2;
    values[kIdxLeftP1] = parameters.leftDistortion.p1;
    values[kIdxLeftP2] = parameters.leftDistortion.p2;
    values[kIdxLeftK3] = parameters.leftDistortion.k3;
    values[kIdxLeftK4] = parameters.leftDistortion.k4;
    values[kIdxLeftK5] = parameters.leftDistortion.k5;
    values[kIdxLeftK6] = parameters.leftDistortion.k6;

    values[kIdxRightFx] = parameters.rightIntrinsics.fx;
    values[kIdxRightFy] = parameters.rightIntrinsics.fy;
    values[kIdxRightCx] = parameters.rightIntrinsics.cx;
    values[kIdxRightCy] = parameters.rightIntrinsics.cy;
    values[kIdxRightSkew] = parameters.rightIntrinsics.skew;

    values[kIdxRightK1] = parameters.rightDistortion.k1;
    values[kIdxRightK2] = parameters.rightDistortion.k2;
    values[kIdxRightP1] = parameters.rightDistortion.p1;
    values[kIdxRightP2] = parameters.rightDistortion.p2;
    values[kIdxRightK3] = parameters.rightDistortion.k3;
    values[kIdxRightK4] = parameters.rightDistortion.k4;
    values[kIdxRightK5] = parameters.rightDistortion.k5;
    values[kIdxRightK6] = parameters.rightDistortion.k6;

    values[kIdxRotation00] = parameters.rotation[0][0];
    values[kIdxRotation01] = parameters.rotation[0][1];
    values[kIdxRotation02] = parameters.rotation[0][2];
    values[kIdxRotation10] = parameters.rotation[1][0];
    values[kIdxRotation11] = parameters.rotation[1][1];
    values[kIdxRotation12] = parameters.rotation[1][2];
    values[kIdxRotation20] = parameters.rotation[2][0];
    values[kIdxRotation21] = parameters.rotation[2][1];
    values[kIdxRotation22] = parameters.rotation[2][2];

    values[kIdxTranslationX] = parameters.translation[0];
    values[kIdxTranslationY] = parameters.translation[1];
    values[kIdxTranslationZ] = parameters.translation[2];

    return values;
}

auto PopulateStereoCameraParameters(const std::array<double, kStereoCameraParametersElementCount> &values, CalibrationParameters &parameters) noexcept -> void
{
    parameters.leftIntrinsics.fx = values[kIdxLeftFx];
    parameters.leftIntrinsics.fy = values[kIdxLeftFy];
    parameters.leftIntrinsics.cx = values[kIdxLeftCx];
    parameters.leftIntrinsics.cy = values[kIdxLeftCy];
    parameters.leftIntrinsics.skew = values[kIdxLeftSkew];

    parameters.leftDistortion.k1 = values[kIdxLeftK1];
    parameters.leftDistortion.k2 = values[kIdxLeftK2];
    parameters.leftDistortion.p1 = values[kIdxLeftP1];
    parameters.leftDistortion.p2 = values[kIdxLeftP2];
    parameters.leftDistortion.k3 = values[kIdxLeftK3];
    parameters.leftDistortion.k4 = values[kIdxLeftK4];
    parameters.leftDistortion.k5 = values[kIdxLeftK5];
    parameters.leftDistortion.k6 = values[kIdxLeftK6];

    parameters.rightIntrinsics.fx = values[kIdxRightFx];
    parameters.rightIntrinsics.fy = values[kIdxRightFy];
    parameters.rightIntrinsics.cx = values[kIdxRightCx];
    parameters.rightIntrinsics.cy = values[kIdxRightCy];
    parameters.rightIntrinsics.skew = values[kIdxRightSkew];

    parameters.rightDistortion.k1 = values[kIdxRightK1];
    parameters.rightDistortion.k2 = values[kIdxRightK2];
    parameters.rightDistortion.p1 = values[kIdxRightP1];
    parameters.rightDistortion.p2 = values[kIdxRightP2];
    parameters.rightDistortion.k3 = values[kIdxRightK3];
    parameters.rightDistortion.k4 = values[kIdxRightK4];
    parameters.rightDistortion.k5 = values[kIdxRightK5];
    parameters.rightDistortion.k6 = values[kIdxRightK6];

    parameters.rotation[0][0] = values[kIdxRotation00];
    parameters.rotation[0][1] = values[kIdxRotation01];
    parameters.rotation[0][2] = values[kIdxRotation02];
    parameters.rotation[1][0] = values[kIdxRotation10];
    parameters.rotation[1][1] = values[kIdxRotation11];
    parameters.rotation[1][2] = values[kIdxRotation12];
    parameters.rotation[2][0] = values[kIdxRotation20];
    parameters.rotation[2][1] = values[kIdxRotation21];
    parameters.rotation[2][2] = values[kIdxRotation22];

    parameters.translation[0] = values[kIdxTranslationX];
    parameters.translation[1] = values[kIdxTranslationY];
    parameters.translation[2] = values[kIdxTranslationZ];
}

// Write calibration data to stream
[[nodiscard]] auto WriteCalibrationData(std::ostream &stream, const CalibrationParameters &parameters) -> bool
{
    stream.write(reinterpret_cast<const char *>(kFileMagic.data()), static_cast<std::streamsize>(kFileMagic.size()));
    if (!stream.good())
    {
        return false;
    }

    if (!WriteUint32(stream, kFileFormatVersion))
    {
        return false;
    }

    const auto payload = FlattenStereoCameraParameters(parameters);
    if (!WriteDoubleArray(stream, payload))
    {
        return false;
    }

    if (!WriteUint32(stream, parameters.imageWidth))
    {
        return false;
    }
    if (!WriteUint32(stream, parameters.imageHeight))
    {
        return false;
    }

    if (!WriteDouble(stream, parameters.reprojectionError))
    {
        return false;
    }

    if (!WriteUint64(stream, parameters.calibrationTime))
    {
        return false;
    }

    if (!WriteCharArray(stream, parameters.leftCameraSerial))
    {
        return false;
    }

    if (!WriteCharArray(stream, parameters.rightCameraSerial))
    {
        return false;
    }

    return true;
}

// Cleanup temporary file on failure
[[nodiscard]] auto CleanupAndFail(const std::filesystem::path &tempPath) noexcept -> Status
{
    RemovePathIfExists(tempPath);
    return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
}

// Determine if any bytes remain unread in the stream.
[[nodiscard]] auto HasTrailingContent(std::istream &stream) -> bool
{
    const auto next = stream.peek();
    if (next == std::char_traits<char>::eof())
    {
        stream.clear(stream.rdstate() & ~std::ios::failbit);
        return false;
    }

    return true;
}

// Read and validate calibration data from stream
[[nodiscard]] auto ReadCalibrationData(std::istream &stream, CalibrationParameters &parameters) -> bool
{
    std::array<std::uint8_t, kFileMagicElementCount> magic{};
    stream.read(reinterpret_cast<char *>(magic.data()), static_cast<std::streamsize>(magic.size()));
    if (!stream || magic != kFileMagic)
    {
        return false;
    }

    std::uint32_t version{};
    if (!ReadUint32(stream, version) || version != kFileFormatVersion)
    {
        return false;
    }

    std::array<double, kStereoCameraParametersElementCount> payload{};
    if (!ReadDoubleArray(stream, payload))
    {
        return false;
    }

    std::uint32_t imageWidth{};
    if (!ReadUint32(stream, imageWidth))
    {
        return false;
    }
    std::uint32_t imageHeight{};
    if (!ReadUint32(stream, imageHeight))
    {
        return false;
    }

    double rms{};
    if (!ReadDouble(stream, rms))
    {
        return false;
    }
    std::uint64_t calibrationTime{};
    if (!ReadUint64(stream, calibrationTime))
    {
        return false;
    }

    std::array<char, kCameraSerialSize> leftSerial{};
    if (!ReadCharArray(stream, leftSerial))
    {
        return false;
    }

    std::array<char, kCameraSerialSize> rightSerial{};
    if (!ReadCharArray(stream, rightSerial))
    {
        return false;
    }

    if (HasTrailingContent(stream))
    {
        return false;
    }

    PopulateStereoCameraParameters(payload, parameters);
    parameters.reprojectionError = rms;
    parameters.calibrationTime = calibrationTime;
    parameters.leftCameraSerial = leftSerial;
    parameters.rightCameraSerial = rightSerial;
    parameters.imageWidth = imageWidth;
    parameters.imageHeight = imageHeight;

    return true;
}
} // namespace

auto SaveCalibrationParameters(const char *filename, const CalibrationParameters &parameters) noexcept -> Status
{
    if (filename == nullptr || filename[0] == '\0')
    {
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    std::filesystem::path tempPath;

    try
    {
        const std::filesystem::path targetPath(filename);

        const auto dirStatus = CreateParentDirectories(targetPath);
        if (!dirStatus.IsOK())
        {
            return dirStatus;
        }

        std::error_code fsError;
        tempPath = MakeTemporaryPath(targetPath, fsError);
        if (fsError || tempPath.empty())
        {
            return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
        }

        std::ofstream out(tempPath, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!out.is_open())
        {
            return CleanupAndFail(tempPath);
        }

        if (!WriteCalibrationData(out, parameters))
        {
            return CleanupAndFail(tempPath);
        }

        out.close();
        if (out.fail())
        {
            return CleanupAndFail(tempPath);
        }

        fsError.clear();
        std::filesystem::rename(tempPath, targetPath, fsError);
        if (fsError)
        {
            return CleanupAndFail(tempPath);
        }
        return Status();
    }
    catch (const std::exception &e)
    {
        LogError(e.what());
        RemovePathIfExists(tempPath);
        return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
    }
    catch (...)
    {
        RemovePathIfExists(tempPath);
        return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
    }
}

auto LoadCalibrationParameters(const char *filename, CalibrationParameters &parameters) noexcept -> Status
{
    if (filename == nullptr || filename[0] == '\0')
    {
        return Status(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    }

    try
    {
        const std::filesystem::path path(filename);
        std::error_code sizeError;
        const auto fileSize = std::filesystem::file_size(path, sizeError);
        if (sizeError || fileSize != kExpectedFileSize)
        {
            return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
        }

        std::ifstream in(filename, std::ios::in | std::ios::binary);
        if (!in.is_open())
        {
            return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
        }

        CalibrationParameters parsed{};
        if (!ReadCalibrationData(in, parsed))
        {
            return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
        }

        parameters = parsed;

        return Status();
    }
    catch (const std::exception &e)
    {
        LogError(e.what());
        return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
    }
    catch (...)
    {
        return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
    }
}
} // namespace retinify
