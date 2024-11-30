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

static void toggle_widget_visibility(GtkButton *button, gpointer widget) {
    bool visible = gtk_widget_get_visible(GTK_WIDGET(widget));
    if(visible) {
        gtk_widget_set_visible(GTK_WIDGET(widget), FALSE);
    } else {
        gtk_widget_set_visible(GTK_WIDGET(widget), TRUE);
    }
}