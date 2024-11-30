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
#include <format/format.hpp>
#include <gtk/gtk.h>
#include <retinify/pipeline.hpp>
#define retinify_get_gui_console retinify::ConsoleContext::Instance()
#define retinify_get_gui_inference retinify::InferenceContext::Instance()
#define retinify_get_gui_calibration retinify::CalibrationContext::Instance()
namespace retinify
{
class InferenceContext : public Context<InferenceContext>
{
  public:
    InferenceContext();
    ~InferenceContext();

  private:
    std::unique_ptr<retinify::Expander> inference_expander;
    std::unique_ptr<retinify::Button> inference_button;
    std::unique_ptr<retinify::Button> camera_button;
    std::unique_ptr<retinify::Button> loader_button;
};

class CalibrationContext : public Context<CalibrationContext>
{
  public:
    CalibrationContext();
    ~CalibrationContext();

  private:
    std::unique_ptr<retinify::DropDown> image_size_dropdown;

    // pattern
    std::unique_ptr<retinify::Expander> pattern_expander;
    std::unique_ptr<retinify::DropDown> pattern_drop_down;

    // device
    std::unique_ptr<retinify::Expander> device_expander;
    std::unique_ptr<retinify::Grid> device_grid;
    std::unique_ptr<retinify::DropDown> left_id_drop_down;
    std::unique_ptr<retinify::DropDown> right_id_drop_down;

    // calibration
    std::unique_ptr<retinify::Expander> calibration_expander;
    std::unique_ptr<retinify::Grid> calibration_grid;
    std::unique_ptr<retinify::Button> calibration_button;

    // test
    std::unique_ptr<retinify::Expander> test_expander;
    std::unique_ptr<retinify::Grid> test_grid;
};

class ConsoleContext : public Context<ConsoleContext>
{
  public:
    ConsoleContext();
    ~ConsoleContext();

  private:
    std::unique_ptr<retinify::ScrollWindow> scroll_;
    std::unique_ptr<retinify::List> list_;

    std::unique_ptr<retinify::ScrollWindow> area1_scroll_;
    std::unique_ptr<retinify::ScrollWindow> area2_scroll_;
    std::unique_ptr<retinify::ScrollWindow> area3_scroll_;
};
} // namespace retinify