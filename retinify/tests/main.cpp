// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-eula

#include "retinify/logging.hpp"

#include <gtest/gtest.h>

int main(int argc, char **argv)
{
    retinify::SetLogLevel(retinify::LogLevel::DEBUG);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}