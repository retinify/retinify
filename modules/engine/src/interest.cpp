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

#include <algorithm>
#include <retinify/interest.hpp>

retinify::Interest::Interest() = default;

retinify::Interest::Interest(cv::Size size)
{
    this->SetImageSize(size);
}

retinify::Interest::~Interest() = default;

void retinify::Interest::SetImageSize(cv::Size size)
{
    this->size_ = size;
}

void retinify::Interest::SetScaledRectArray(float x_scale, float y_scale, float w_scale, float h_scale,
                                            Interest::Mode mode)
{
    x_scale = std::clamp(x_scale, 0.0f, 1.0f);
    y_scale = std::clamp(y_scale, 0.0f, 1.0f);
    w_scale = std::clamp(w_scale, 0.0f, 1.0f);
    h_scale = std::clamp(h_scale, 0.0f, 1.0f);

    int width = this->size_.width * w_scale;
    int height = this->size_.height * h_scale;
    int x = (1 - w_scale) * this->size_.width * x_scale;
    int y = (1 - h_scale) * this->size_.height * y_scale;

    this->rect_[0] = cv::Rect(x, y, width, height);
    this->rect_[1] = cv::Rect(x, y, width, height);
}

std::array<cv::Rect, 2> retinify::Interest::GetRectArray() const
{
    return this->rect_;
}