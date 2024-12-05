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

retinify::Hub::Hub()
{
    this->pipeline_ = std::make_unique<Pipeline>();
}

retinify::Hub::~Hub()
{
}

bool retinify::Hub::IsHubRunning()
{
    return this->running_;
}

void retinify::Hub::ActivateHub()
{
    this->running_ = true;
}

void retinify::Hub::DeactivateHub()
{
    this->running_ = false;
    this->pipeline_->Stop();
}

std::unique_ptr<retinify::Pipeline> &retinify::Hub::GetPipelinePtr()
{
    return this->pipeline_;
}