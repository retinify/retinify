// Copyright (C) 2024 retinify project team. All rights reserved.
//
// This file is part of retinify.
//
// retinify is free software: you can redistribute it and/or modify it under the terms of the
// GNU Affero General Public License as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// retinify is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with retinify.
// If not, see <https://www.gnu.org/licenses/>.

#pragma once
#include <array>
#include <opencv2/core.hpp>
#include <retinify/mutex.hpp>
namespace retinify
{
class Interest
{
  public:
    Interest();
    Interest(cv::Size size);
    ~Interest();
    enum class Mode
    {
        SAME,
        MIRROR,
    };
    void SetImageSize(cv::Size size);
    void SetScaledRectArray(float x_scale, float y_scale, float w_scale, float h_scale,
                            Interest::Mode mode = Interest::Mode::SAME);
    std::array<cv::Rect, 2> GetRectArray() const;

  private:
    retinify::Mutex mutex_;
    cv::Size size_;
    std::array<cv::Rect, 2> rect_;
};
} // namespace retinify