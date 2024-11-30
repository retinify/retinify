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

#include <retinify/pipeline.hpp>

retinify::Core::Core()
{
    this->pipeline_ = std::make_unique<Pipeline>();
}

retinify::Core::~Core()
{
}

bool retinify::Core::IsCoreRunning()
{
    return this->running_;
}

void retinify::Core::ActivateCore()
{
    this->running_ = true;
}

void retinify::Core::DeactivateCore()
{
    this->running_ = false;
    this->pipeline_->Stop();
}

std::unique_ptr<retinify::Pipeline> &retinify::Core::GetPipelinePtr()
{
    return this->pipeline_;
}