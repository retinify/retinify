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
#include <console/console.hpp>
#include <glviewer/glviewer.hpp>
#include <gtk/gtk.h>
#include <gupdater.hpp>
#include <gutils.hpp>
#include <image/image.hpp>
#include <retinify/core.hpp>
#include <sidebar/sidebar.hpp>

#define RETINIFY_WINDOW_HEIGHT 600
#define RETINIFY_WINDOW_WIDTH 1200

namespace retinify
{
inline static void InitProvider()
{
    GtkCssProvider *provider = gtk_css_provider_new();
    gtk_css_provider_load_from_path(provider, RETINIFY_GUI_STYLE_CSS_PATH.c_str());
    gtk_style_context_add_provider_for_display(gdk_display_get_default(), GTK_STYLE_PROVIDER(provider),
                                               GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    g_object_set(gtk_settings_get_default(), "gtk-application-prefer-dark-theme", TRUE, NULL);

    g_object_unref(provider);
}

inline static void CloseWindow(GtkWidget &widget, gpointer user_data)
{
    retinify_get_hub.DeactivateHub();
}

inline static void ActivateWindow(GtkApplication &app, gpointer user_data)
{
    GtkWidget *window;
    window = gtk_application_window_new(&app);

    std::unique_ptr<retinify::Horizontal> main_box = std::make_unique<retinify::Horizontal>();

    GtkWidget *middle_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_box_append(GTK_BOX(middle_box), retinify_get_gui_image.Get());
    gtk_box_append(GTK_BOX(middle_box), retinify_get_gui_glviewer.Get());

    // main_box->Append(retinify_get_gui_sidebar.Get());
    main_box->Append(middle_box);
    main_box->Append(retinify_get_gui_console.Get());

    // GtkHeaderBarを作成
    GtkWidget *header_bar = gtk_header_bar_new();
    GtkWidget *title_widget = gtk_label_new("retinify");
    std::unique_ptr<retinify::Button> tab_console_button = std::make_unique<retinify::Button>("");

    gtk_header_bar_set_show_title_buttons(GTK_HEADER_BAR(header_bar), TRUE);
    gtk_header_bar_set_title_widget(GTK_HEADER_BAR(header_bar), title_widget);
    gtk_header_bar_pack_end(GTK_HEADER_BAR(header_bar), tab_console_button->Get());

    // signal
    g_signal_connect(tab_console_button->Get(), "clicked", G_CALLBACK(toggle_widget_visibility),
                     retinify_get_gui_console.Get());

    gtk_window_set_child(GTK_WINDOW(window), main_box->Get());
    gtk_window_set_titlebar(GTK_WINDOW(window), header_bar);
    gtk_window_set_default_size(GTK_WINDOW(window), RETINIFY_WINDOW_WIDTH, RETINIFY_WINDOW_HEIGHT);

    g_signal_connect(window, "destroy", G_CALLBACK(CloseWindow), nullptr);

    // provider
    retinify::InitProvider();

    // updater
    retinify::Updater updater;
    g_timeout_add(1000 / 60, updater.Update, nullptr);
    gtk_window_present(GTK_WINDOW(window));
}
} // namespace retinify