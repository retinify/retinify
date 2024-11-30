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
#include <gtk/gtk.h>
#include <retinify/pipeline.hpp>
#define retinify_get_gui_image retinify::ImageBox::Instance()
namespace retinify
{
class ImageBox : public Context<ImageBox>
{
  public:
    ImageBox();
    ~ImageBox() = default;
    void UpdateDisplayStereoData(StereoImageData &data);

  private:
    GtkWidget *image_box{nullptr};
    cv::Size image_size{320, 180};
    GtkWidget *left_image_widget{nullptr};
    GtkWidget *right_image_widget{nullptr};
    GtkWidget *image3{nullptr};
    GtkWidget *image4{nullptr};
};
} // namespace retinify