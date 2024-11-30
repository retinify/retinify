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

#include <sidebar/sidebar.hpp>

retinify::SideBar::SideBar()
{
    GtkWidget *side_button_1 = gtk_button_new();
    GtkWidget *side_button_2 = gtk_button_new();
    GtkWidget *side_button_3 = gtk_button_new();

    this->Append(side_button_1);
    this->Append(side_button_2);
    this->Append(side_button_3);

    gtk_widget_add_css_class(side_button_1, "side-button");
    gtk_widget_add_css_class(side_button_2, "side-button");
    gtk_widget_add_css_class(side_button_3, "side-button");
}

retinify::SideBar::~SideBar() = default;