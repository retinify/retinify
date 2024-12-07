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

#include <console/console.hpp>
#include <glviewer/glviewer.hpp>
#include <gtk/gtk.h>
#include <retinify/pipeline.hpp>
#include <retinify/core.hpp>

inline static void OnCameraButtonClicked(GtkButton *button, gpointer user_data)
{
    retinify::CalibrationData calib_data;
    if (!calib_data.Read(RETINIFY_DEFAULT_CALIBRATION_FILE_PATH))
    {
        std::cerr << "Error: Configuration file not found" << std::endl;
        return;
    }
    retinify_get_hub.GetPipelinePtr()->Start(calib_data, retinify::Pipeline::Mode::RECTIFY); /// @todo change to RAWIMAGE
}

inline static void OnInfernceButtonClicked(GtkButton *button, gpointer user_data)
{
    retinify::CalibrationData calib_data;
    if (!calib_data.Read(RETINIFY_DEFAULT_CALIBRATION_FILE_PATH))
    {
        std::cerr << "Error: Configuration file not found" << std::endl;
        return;
    }
    retinify_get_hub.GetPipelinePtr()->Start(calib_data, retinify::Pipeline::Mode::INFERENCE);
}

inline static void OnLoaderButtonClicked(GtkButton *button, gpointer user_data)
{
    retinify::CalibrationData calib_data;
    if (!calib_data.Read(RETINIFY_DEFAULT_CALIBRATION_FILE_PATH))
    {
        std::cerr << "Error: Configuration file not found" << std::endl;
        return;
    }
    retinify_get_hub.GetPipelinePtr()->Start(calib_data, retinify::Pipeline::Mode::LOADER);
}

retinify::InferenceContext::InferenceContext()
{
    // Create widgets
    GtkWidget *spin_button = gtk_spin_button_new_with_range(0, 100, 1);
    this->camera_button = std::make_unique<retinify::Button>("Camera");
    this->inference_button = std::make_unique<retinify::Button>("Inference");
    this->loader_button = std::make_unique<retinify::Button>("Loader");
    this->inference_expander = std::make_unique<retinify::Expander>("Inference");

    this->Append(spin_button);
    this->Append(camera_button->Get());
    this->Append(inference_button->Get());
    this->Append(loader_button->Get());
    this->Append(inference_expander->Get());

    // Connect signals
    g_signal_connect(camera_button->Get(), "clicked", G_CALLBACK(OnCameraButtonClicked), this);
    g_signal_connect(inference_button->Get(), "clicked", G_CALLBACK(OnInfernceButtonClicked), this);
    g_signal_connect(loader_button->Get(), "clicked", G_CALLBACK(OnLoaderButtonClicked), this);
}

retinify::InferenceContext::~InferenceContext() = default;