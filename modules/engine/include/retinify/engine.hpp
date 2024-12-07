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
#include <retinify/core.hpp>
#include <vector>
namespace retinify
{
class StereoEngine
{
public:
    StereoEngine();
    ~StereoEngine();

    void SetInputQueue(retinify::Queue<retinify::StereoImageData> *queue);
    void SetOutputQueue(retinify::Queue<retinify::StereoImageData> *queue);

    enum class Mode
    {
        RAWIMAGE,
        RECTIFY,
        CALIBRATION,
        INFERENCE,
    };
    void Start(CalibrationData &calib, StereoEngine::Mode mode);
    void Stop();

    StereoEngine(const StereoEngine &) = delete;
    StereoEngine &operator=(const StereoEngine &) = delete;

    StereoEngine(StereoEngine &&) noexcept = default;
    StereoEngine &operator=(StereoEngine &&) noexcept = default;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace retinify