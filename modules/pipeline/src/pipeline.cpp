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

#include <iostream>
#include <retinify/core.hpp>
#include <retinify/pipeline.hpp>
#include <retinify/engine.hpp>
#include <retinify/io.hpp>
/**
 * Pipeline
 */
retinify::Pipeline::Pipeline()
{
    this->impl_ = std::make_unique<Impl>();
}

retinify::Pipeline::~Pipeline() = default;

class retinify::Pipeline::Impl
{
public:
    Impl()
    {
        camera_.SetOutputQueue(&this->camera_queue_);
        engine_.SetInputQueue(&this->camera_queue_);
        engine_.SetOutputQueue(&this->engine_queue_);
    }
    
    ~Impl()
    {
        this->Stop();
    }

    inline void Stop()
    {
        this->camera_.Stop();
        this->engine_.Stop();
    }

    inline std::unique_ptr<retinify::StereoImageData> GetOutputData()
    {
        return this->engine_queue_.Pop();
    }

    inline std::unique_ptr<retinify::StereoImageData> TryGetOutputData()
    {
        return this->engine_queue_.TryPop();
    }

    inline void Start(retinify::CalibrationData &calib, const retinify::Pipeline::Mode mode)
    {
        this->Stop(); // if already active, deactivate first

        this->camera_.Start(calib);

        switch (mode)
        {
        case retinify::Pipeline::Mode::RAWIMAGE:
            engine_.Start(calib, retinify::StereoEngine::Mode::RAWIMAGE);
            break;

        case retinify::Pipeline::Mode::RECTIFY:
            engine_.Start(calib, retinify::StereoEngine::Mode::RECTIFY);
            break;

        case retinify::Pipeline::Mode::CALIBRATION:
            engine_.Start(calib, retinify::StereoEngine::Mode::CALIBRATION);
            break;

        case retinify::Pipeline::Mode::INFERENCE:
            engine_.Start(calib, retinify::StereoEngine::Mode::INFERENCE);
            break;
        }
    }

private:
    retinify::Queue<retinify::StereoImageData> camera_queue_;
    retinify::Queue<retinify::StereoImageData> engine_queue_;
    retinify::Camera camera_;
    retinify::StereoEngine engine_;
};

void retinify::Pipeline::Stop()
{
    this->impl_->Stop();
}

std::unique_ptr<retinify::StereoImageData> retinify::Pipeline::GetOutputData()
{
    return this->impl_->GetOutputData();
}

std::unique_ptr<retinify::StereoImageData> retinify::Pipeline::TryGetOutputData()
{
    return this->impl_->TryGetOutputData();
}

void retinify::Pipeline::Start(retinify::CalibrationData &calib, const Mode mode)
{
    this->impl_->Start(calib, mode);
}