// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/version.hpp"

#include <cstring>
#include <gtest/gtest.h>
#include <regex>

namespace retinify
{
class VersionTest : public ::testing::Test
{
};

TEST_F(VersionTest, NotNull)
{
    const char *version = retinify::Version();
    ASSERT_NE(version, nullptr) << "Version() returned nullptr.";
}

TEST_F(VersionTest, NotEmpty)
{
    const char *version = retinify::Version();
    ASSERT_GT(std::strlen(version), 0u) << "Version() returned an empty string.";
}

TEST_F(VersionTest, MatchesSemVer)
{
    const char *version = retinify::Version();
    std::regex semverPattern(R"(^\d+\.\d+\.\d+(-[A-Za-z0-9\.\-]+)?(\+[A-Za-z0-9\.\-]+)?$)");
    EXPECT_TRUE(std::regex_match(version, semverPattern)) << "Version string does not match SemVer: " << version;
}
} // namespace retinify
