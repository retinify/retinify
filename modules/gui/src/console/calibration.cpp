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

#include <console/calibration.func.hpp>
#include <console/console.hpp>
retinify::CalibrationContext::CalibrationContext()
{
    // text
    GtkWidget *calibration_label = gtk_label_new("Calibration");
    gtk_widget_set_halign(calibration_label, GTK_ALIGN_START);

    this->Append(calibration_label);

    GtkWidget *left_label = gtk_label_new("Left");
    GtkWidget *right_label = gtk_label_new("Right");
    GtkWidget *left_id_label = gtk_label_new("id");
    GtkWidget *right_id_label = gtk_label_new("id");

    this->image_size_dropdown = std::make_unique<retinify::DropDown>();
    this->image_size_dropdown->AppendText(IMAGE_SIZE_640_480);
    this->image_size_dropdown->AppendText(IMAGE_SIZE_1280_720);
    this->image_size_dropdown->SetActive(1);

    // pattern
    this->pattern_expander = std::make_unique<retinify::Expander>("Pattern");

    this->Append(pattern_expander->Get());

    // device
    this->device_expander = std::make_unique<retinify::Expander>("Device");

    this->device_grid = std::make_unique<retinify::Grid>();
    this->left_id_drop_down = std::make_unique<retinify::DropDown>();
    this->right_id_drop_down = std::make_unique<retinify::DropDown>();
    this->calibration_button = std::make_unique<retinify::Button>("Calibration");

    this->left_id_drop_down->AppendText("000000000000000");
    this->left_id_drop_down->AppendText("1");
    this->left_id_drop_down->AppendText("2");
    this->right_id_drop_down->AppendText("0");
    this->right_id_drop_down->AppendText("1");
    this->right_id_drop_down->AppendText("2");

    this->device_grid->Attach(left_label, 0, 0, 1, 1);
    this->device_grid->Attach(right_label, 0, 1, 1, 1);
    this->device_grid->Attach(left_id_label, 1, 0, 1, 1);
    this->device_grid->Attach(right_id_label, 1, 1, 1, 1);
    this->device_grid->Attach(this->left_id_drop_down->Get(), 2, 0, 1, 1);
    this->device_grid->Attach(this->right_id_drop_down->Get(), 2, 1, 1, 1);
    this->device_grid->Attach(image_size_dropdown->Get(), 0, 2, 3, 1);
    this->device_grid->Attach(calibration_button->Get(), 0, 3, 3, 1);

    device_expander->Append(this->device_grid->Get());
    // device_expander->SetExpanded(true);

    this->Append(device_expander->Get());

    // calibration
    this->calibration_expander = std::make_unique<retinify::Expander>("Calibration");
    this->Append(this->calibration_expander->Get());

    // test
    this->test_expander = std::make_unique<retinify::Expander>("Test");
    this->Append(this->test_expander->Get());

    // Signals
    g_signal_connect(image_size_dropdown->Get(), "notify::selected", G_CALLBACK(OnImageSizeChanged), this);
    g_signal_connect(calibration_button->Get(), "clicked", G_CALLBACK(OnCalibrationButtonClicked), this);
}

retinify::CalibrationContext::~CalibrationContext() = default;