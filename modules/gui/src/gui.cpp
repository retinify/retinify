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

#include <retinify/gui.hpp>
#include <activate.hpp>

int retinify::gmain()
{
    GtkApplication *app = gtk_application_new("com.github.retinify", G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect(app, "activate", G_CALLBACK(retinify::ActivateWindow), nullptr);
    int status = g_application_run(G_APPLICATION(app), 0, nullptr);
    g_object_unref(app);
    return status;
}