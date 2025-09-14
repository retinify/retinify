// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/logging.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <source_location>
#include <sstream>
#include <string>

namespace retinify
{
class StdoutCapture
{
  public:
    void Start()
    {
        old_buf_ = std::cout.rdbuf(buffer_.rdbuf());
    }
    std::string Stop()
    {
        std::cout.rdbuf(old_buf_);
        return buffer_.str();
    }

  private:
    std::ostringstream buffer_;
    std::streambuf *old_buf_{nullptr};
};

class StderrCapture
{
  public:
    void Start()
    {
        old_buf_ = std::cerr.rdbuf(buffer_.rdbuf());
    }
    std::string Stop()
    {
        std::cerr.rdbuf(old_buf_);
        return buffer_.str();
    }

  private:
    std::ostringstream buffer_;
    std::streambuf *old_buf_{nullptr};
};

class LogTest : public ::testing::Test
{
};

TEST(LogTest, LogDebug)
{
    StdoutCapture cap;
    SetLogLevel(LogLevel::DEBUG);

    cap.Start();
    LogDebug("Debug via cout");
    std::string out = cap.Stop();

    ASSERT_NE(out.find("Debug via cout"), std::string::npos);
}

TEST(LogTest, LogInfo)
{
    StdoutCapture cap;
    SetLogLevel(LogLevel::INFO);

    cap.Start();
    LogInfo("Info via cout");
    std::string out = cap.Stop();

    ASSERT_NE(out.find("Info via cout"), std::string::npos);
}

TEST(LogTest, LogWarn)
{
    StderrCapture cap;
    SetLogLevel(LogLevel::INFO);

    cap.Start();
    LogWarn("Warning via cerr");
    std::string out = cap.Stop();

    ASSERT_NE(out.find("Warning via cerr"), std::string::npos);
}

TEST(LogTest, LogError)
{
    StderrCapture cap;
    SetLogLevel(LogLevel::INFO);

    cap.Start();
    LogError("Error via cerr");
    std::string out = cap.Stop();

    ASSERT_NE(out.find("Error via cerr"), std::string::npos);
}

TEST(LogTest, LogFatal)
{
    StderrCapture cap;
    SetLogLevel(LogLevel::INFO);

    cap.Start();
    LogFatal("Fatal via cerr");
    std::string out = cap.Stop();

    ASSERT_NE(out.find("Fatal via cerr"), std::string::npos);
}
} // namespace retinify