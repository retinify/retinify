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
#include <image/image.hpp>

class PipelineControl
{
public:
    PipelineControl(const char *label)
    {
        this->vertical_box = std::make_unique<retinify::VerticalBox>();
        this->label = std::make_unique<retinify::Label>(label);

        this->grid = std::make_unique<retinify::Grid>();
        this->l_label = std::make_unique<retinify::Label>("L");
        this->r_label = std::make_unique<retinify::Label>("R");
        this->l_drop_down = std::make_unique<retinify::DropDown>();
        this->r_drop_down = std::make_unique<retinify::DropDown>();

        this->l_drop_down->AppendText("Unit1");
        this->l_drop_down->AppendText("Unit2");
        this->r_drop_down->AppendText("Unit1");
        this->r_drop_down->AppendText("Unit2");

        this->grid->Attach(this->l_label->Get(), 0, 0, 1, 1);
        this->grid->Attach(this->r_label->Get(), 0, 1, 1, 1);
        this->grid->Attach(this->l_drop_down->Get(), 1, 0, 1, 1);
        this->grid->Attach(this->r_drop_down->Get(), 1, 1, 1, 1);

        this->vertical_box->Append(this->label->Get());
        this->vertical_box->Append(this->grid->Get());
    }
    ~PipelineControl() = default;

    GtkWidget *Get()
    {
        return this->vertical_box->Get();
    }

private:
    std::unique_ptr<retinify::VerticalBox> vertical_box;

    std::unique_ptr<retinify::Label> label;
    std::unique_ptr<retinify::Grid> grid;

    std::unique_ptr<retinify::Label> l_label;
    std::unique_ptr<retinify::Label> r_label;

    std::unique_ptr<retinify::DropDown> l_drop_down;
    std::unique_ptr<retinify::DropDown> r_drop_down;
};

void static HandlePipelineMode(GtkWidget *widget, gpointer data)
{
    auto *switch_widget = GTK_SWITCH(widget);
    if (gtk_switch_get_active(switch_widget))
    {
        retinify_get_gui_inference.SetVisible(false);
        retinify_get_gui_calibration.SetVisible(true);
    }
    else
    {
        retinify_get_gui_inference.SetVisible(true);
        retinify_get_gui_calibration.SetVisible(false);
    }
}

retinify::ConsoleContext::ConsoleContext()
{
    /**
     * UPPER AREA
     */
    // text
    std::unique_ptr<retinify::VerticalBox> upper_area = std::make_unique<retinify::VerticalBox>();

    std::unique_ptr<retinify::Label> label = std::make_unique<retinify::Label>("Calibration");
    std::unique_ptr<retinify::Horizontal> horizontal = std::make_unique<retinify::Horizontal>(5);
    gtk_widget_set_halign(horizontal->Get(), GTK_ALIGN_END);

    std::unique_ptr<retinify::Switch> calibration_switch = std::make_unique<retinify::Switch>();
    g_signal_connect(calibration_switch->Get(), "notify::active", G_CALLBACK(HandlePipelineMode), nullptr);

    horizontal->Append(label->Get());
    horizontal->Append(calibration_switch->Get());

    // scroll
    this->scroll_ = std::make_unique<retinify::ScrollWindow>();

    // list
    this->list_ = std::make_unique<retinify::List>();
    std::unique_ptr<PipelineControl> pselect1 = std::make_unique<PipelineControl>("Pipeline 1");
    this->list_->Append(pselect1->Get());

    this->scroll_->Append(this->list_->Get());
    upper_area->Append(horizontal->Get());
    upper_area->Append(this->scroll_->Get());

    /**
     * CONSOLE AREA
     */
    GtkWidget *console_area = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    this->area1_scroll_ = std::make_unique<retinify::ScrollWindow>();
    gtk_box_append(GTK_BOX(console_area), this->area1_scroll_->Get());
    this->area1_scroll_->Append(retinify_get_gui_inference.Get());
    this->area1_scroll_->Append(retinify_get_gui_calibration.Get());
    retinify_get_gui_calibration.SetVisible(false);

    // pane
    GtkWidget *pane = gtk_paned_new(GTK_ORIENTATION_VERTICAL);
    gtk_paned_set_start_child(GTK_PANED(pane), upper_area->Get());
    gtk_paned_set_end_child(GTK_PANED(pane), console_area);
    gtk_paned_set_position(GTK_PANED(pane), 200);

    this->Append(pane);

    // default
    calibration_switch->SetActive(true);
}

retinify::ConsoleContext::~ConsoleContext() = default;