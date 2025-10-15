// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: LicenseRef-retinify-eula

#include "retinify/status.hpp"

#include <gtest/gtest.h>

namespace retinify
{
class StatusTest : public ::testing::Test
{
};

TEST_F(StatusTest, DefaultConstructor)
{
    Status status;
    EXPECT_TRUE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::NONE);
    EXPECT_EQ(status.Code(), StatusCode::OK);
}

TEST_F(StatusTest, ParameterizedConstructor_OK)
{
    Status status{StatusCategory::RETINIFY, StatusCode::OK};
    EXPECT_TRUE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::RETINIFY);
    EXPECT_EQ(status.Code(), StatusCode::OK);
}

TEST_F(StatusTest, ParameterizedConstructor_Fail)
{
    Status status{StatusCategory::SYSTEM, StatusCode::FAIL};
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(status.Category(), StatusCategory::SYSTEM);
    EXPECT_EQ(status.Code(), StatusCode::FAIL);
}

TEST_F(StatusTest, ParameterizedConstructor_InvalidArgument)
{
    Status s(StatusCategory::USER, StatusCode::INVALID_ARGUMENT);
    EXPECT_FALSE(s.IsOK());
    EXPECT_EQ(s.Category(), StatusCategory::USER);
    EXPECT_EQ(s.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(StatusTest, CopyConstructorAndAssignment)
{
    Status status1(StatusCategory::CUDA, StatusCode::FAIL);
    Status status2 = status1;
    EXPECT_EQ(status2.Category(), StatusCategory::CUDA);
    EXPECT_EQ(status2.Code(), StatusCode::FAIL);

    Status status3;
    status3 = status1;
    EXPECT_EQ(status3.Category(), StatusCategory::CUDA);
    EXPECT_EQ(status3.Code(), StatusCode::FAIL);
}

TEST_F(StatusTest, MoveConstructorAndAssignment)
{
    Status status1(StatusCategory::RETINIFY, StatusCode::OK);
    Status status2 = std::move(status1);
    EXPECT_EQ(status2.Category(), StatusCategory::RETINIFY);
    EXPECT_EQ(status2.Code(), StatusCode::OK);

    Status status3;
    status3 = std::move(status2);
    EXPECT_EQ(status3.Category(), StatusCategory::RETINIFY);
    EXPECT_EQ(status3.Code(), StatusCode::OK);
}
} // namespace retinify
