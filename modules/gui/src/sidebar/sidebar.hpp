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
#include <retinify/pipeline.hpp>
#include <format/format.hpp>
#define retinify_get_gui_sidebar retinify::SideBar::Instance()
namespace retinify
{
class SideBar : public Context<SideBar>
{
  public:
    SideBar();
    ~SideBar();
};
} // namespace retinify