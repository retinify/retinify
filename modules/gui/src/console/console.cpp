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

class PipelineSelect
{
  public:
    PipelineSelect(const char *label)
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
    ~PipelineSelect() = default;

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

retinify::ConsoleContext::ConsoleContext()
{
    /**
     * UPPER AREA
     */
    // text
    GtkWidget *upper_area = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    GtkWidget *pipeline_label = gtk_label_new("Pipeline");
    gtk_widget_set_halign(pipeline_label, GTK_ALIGN_START);

    // scroll
    this->scroll_ = std::make_unique<retinify::ScrollWindow>();

    // list
    this->list_ = std::make_unique<retinify::List>();
    std::unique_ptr<PipelineSelect> pselect1 = std::make_unique<PipelineSelect>("1");
    std::unique_ptr<PipelineSelect> pselect2 = std::make_unique<PipelineSelect>("2");

    this->list_->Append(pselect1->Get());
    this->list_->Append(pselect2->Get());

    this->scroll_->Append(pipeline_label);
    this->scroll_->Append(this->list_->Get());

    gtk_box_append(GTK_BOX(upper_area), this->scroll_->Get());

    /**
     * CONSOLE AREA
     */
    GtkWidget *console_area = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    GtkWidget *sidebar = gtk_stack_sidebar_new();
    GtkWidget *stack = gtk_stack_new();
    gtk_stack_sidebar_set_stack(GTK_STACK_SIDEBAR(sidebar), GTK_STACK(stack));
    gtk_box_append(GTK_BOX(console_area), sidebar);
    gtk_box_append(GTK_BOX(console_area), stack);
    gtk_widget_set_vexpand(console_area, TRUE);
    GtkWidget *area1 = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    GtkWidget *area2 = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    GtkWidget *area3 = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_stack_add_titled(GTK_STACK(stack), area1, "page1", "");
    gtk_stack_add_titled(GTK_STACK(stack), area2, "page2", "");
    gtk_stack_add_titled(GTK_STACK(stack), area3, "page3", "");
    this->area1_scroll_ = std::make_unique<retinify::ScrollWindow>();
    this->area2_scroll_ = std::make_unique<retinify::ScrollWindow>();
    this->area3_scroll_ = std::make_unique<retinify::ScrollWindow>();

    // area1
    gtk_box_append(GTK_BOX(area1), this->area1_scroll_->Get());
    this->area1_scroll_->Append(retinify_get_gui_inference.Get());

    // area2
    gtk_box_append(GTK_BOX(area2), this->area2_scroll_->Get());
    this->area2_scroll_->Append(retinify_get_gui_calibration.Get());

    // pane
    GtkWidget *pane = gtk_paned_new(GTK_ORIENTATION_VERTICAL);
    gtk_paned_set_start_child(GTK_PANED(pane), upper_area);
    gtk_paned_set_end_child(GTK_PANED(pane), console_area);
    gtk_paned_set_position(GTK_PANED(pane), 200);

    this->Append(pane);
}

retinify::ConsoleContext::~ConsoleContext() = default;