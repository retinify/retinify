// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-EULA

#include "retinify/paths.hpp"

#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>

namespace retinify
{
class PathTest : public ::testing::Test
{
  protected:
    void CheckPath(const char *path, bool shouldBeDirectory)
    {
        ASSERT_NE(path, nullptr) << "Path pointer is nullptr.";
        ASSERT_GT(std::strlen(path), 0u) << "Path string is empty.";

        std::error_code errorCode;
        const std::filesystem::path fsPath(path);

        const bool exists = std::filesystem::exists(fsPath, errorCode);
        ASSERT_FALSE(errorCode) << "Error checking path existence: " << errorCode.message() << " for path: " << path;
        ASSERT_TRUE(exists) << "Path does not exist: " << path;

        const auto status = std::filesystem::status(fsPath, errorCode);
        ASSERT_FALSE(errorCode) << "Error getting file status: " << errorCode.message() << " for path: " << path;

        if (shouldBeDirectory)
        {
            ASSERT_TRUE(std::filesystem::is_directory(status)) << "Path is not a directory: " << path << " (actual type: " << static_cast<int>(status.type()) << ")";
        }
        else
        {
            ASSERT_TRUE(std::filesystem::is_regular_file(status)) << "Path is not a regular file: " << path << " (actual type: " << static_cast<int>(status.type()) << ")";
        }
    }
};

TEST_F(PathTest, HomeDirectoryPath)
{
    CheckPath(HomeDirectoryPath(), true);
}

TEST_F(PathTest, ConfigDirectoryPath)
{
    CheckPath(ConfigDirectoryPath(), true);
}

TEST_F(PathTest, CacheDirectoryPath)
{
    CheckPath(CacheDirectoryPath(), true);
}

TEST_F(PathTest, DataDirectoryPath)
{
    CheckPath(DataDirectoryPath(), true);
}

TEST_F(PathTest, StateDirectoryPath)
{
    CheckPath(StateDirectoryPath(), true);
}

TEST_F(PathTest, ONNXModelFilePath)
{
    CheckPath(ONNXModelFilePath(), false);
}
} // namespace retinify