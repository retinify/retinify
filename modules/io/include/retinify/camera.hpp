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
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <retinify/core.hpp>
namespace retinify
{
class CameraThread
{
public:
    CameraThread();
    ~CameraThread();

    void Start(const char *node1, const char *node2);

    void Stop();

    bool SetOutputQueue(retinify::Queue<retinify::StereoImageData> *queue);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace retinify