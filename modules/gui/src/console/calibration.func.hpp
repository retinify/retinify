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
#include <glviewer/glviewer.hpp>
#define IMAGE_SIZE_640_480 "640 x 480"
#define IMAGE_SIZE_1280_720 "1280 x 720"
inline static void OnImageSizeChanged(GtkDropDown *dropdown, gpointer user_data)
{
    guint selected = gtk_drop_down_get_selected(GTK_DROP_DOWN(dropdown));
    GtkStringList *string_list = GTK_STRING_LIST(gtk_drop_down_get_model(GTK_DROP_DOWN(dropdown)));
    const char *option = gtk_string_list_get_string(string_list, selected);

    if (g_strcmp0(option, IMAGE_SIZE_640_480) == 0)
    {
        int width = 640;
        int height = 480;
        retinify_get_gui_glviewer.ApplyConfiguration(width, height);
        std::cout << "Width: " << width << " Height: " << height << std::endl;
    }
    else if (g_strcmp0(option, IMAGE_SIZE_1280_720) == 0)
    {
        int width = 1280;
        int height = 720;
        retinify_get_gui_glviewer.ApplyConfiguration(width, height);
        std::cout << "Width: " << width << " Height: " << height << std::endl;
    }
    else
    {
        std::cout << "Invalid option" << std::endl;
    }
}

inline static void OnCalibrationButtonClicked(GtkButton *button, gpointer user_data)
{
    retinify::CalibrationData calib_data;
    if (!calib_data.Read(RETINIFY_DEFAULT_CALIBRATION_FILE_PATH))
    {
        std::cerr << "Error: Configuration file not found" << std::endl;
        return;
    }
    retinify_get_hub.GetPipelinePtr()->Start(calib_data, retinify::Pipeline::Mode::CALIBRATION);
}
