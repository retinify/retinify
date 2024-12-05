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
#include <glviewer/glviewer.hpp>
#include <image/image.hpp>

namespace retinify
{
class Updater
{
  public:
    inline static gboolean Update(gpointer user_data)
    {
        auto running = retinify_get_hub.IsHubRunning();
        if (!running)
        {
            return G_SOURCE_REMOVE;
        }

        if (auto frame_data = retinify_get_hub.GetPipelinePtr()->TryGetOutputData())
        {
            std::unique_ptr<retinify::StereoImageData> data = std::move(frame_data);

            cv::cvtColor(data->left_.image_, data->left_.image_, cv::COLOR_BGR2RGB);
            cv::cvtColor(data->right_.image_, data->right_.image_, cv::COLOR_BGR2RGB);

            retinify_get_gui_image.UpdateDisplayStereoData(*data);
            retinify_get_gui_glviewer.UpdatePCDPositionsAndColors(*data);
        }

        return G_SOURCE_CONTINUE;
    }
};
} // namespace retinify