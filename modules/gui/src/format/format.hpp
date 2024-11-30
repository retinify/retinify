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

namespace retinify
{
template <typename T> class Context : public retinify::Singleton<T>
{
  public:
    Context()
    {
        this->main_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
        gtk_widget_add_css_class(this->main_, "retinify_context");
    }

    ~Context() = default;

    void Append(GtkWidget *widget)
    {
        gtk_box_append(GTK_BOX(this->main_), widget);
    }

    GtkWidget *Get()
    {
        return this->main_;
    }

  private:
    GtkWidget *main_;
};

class VerticalBox
{
    public:
        VerticalBox()
        {
            this->box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
            // gtk_widget_add_css_class(this->box_), "retinify_vertical");
        }
    
        virtual ~VerticalBox() = default;
    
        void Append(GtkWidget *widget)
        {
            gtk_box_append(GTK_BOX(this->box_), widget);
        }
    
        GtkWidget *Get()
        {
            return this->box_;
        }
    
    private:
        GtkWidget *box_;
};

class Horizontal
{
  public:
    Horizontal()
    {
        this->box_ = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
        // gtk_widget_add_css_class(this->box_), "retinify_horizontal");
    }

    virtual ~Horizontal() = default;

    void Append(GtkWidget *widget)
    {
        gtk_box_append(GTK_BOX(this->box_), widget);
    }

    GtkWidget *Get()
    {
        return this->box_;
    }

  private:
    GtkWidget *box_;
};

class Label
{
  public:
    Label(const char *text)
    {
        this->label_ = gtk_label_new(text);
        gtk_widget_set_halign(this->label_, GTK_ALIGN_START);
        gtk_widget_add_css_class(this->label_, "retinify_label");
    }

    virtual ~Label() = default;

    GtkWidget *Get()
    {
        return this->label_;
    }

  private:
    GtkWidget *label_;
};

class Button
{
  public:
    Button(const char *label)
    {
        this->button_ = gtk_button_new_with_label(label);
        // gtk_widget_set_hexpand(this->button_, true);
        gtk_widget_add_css_class(this->button_, "retinify_button");
    }

    virtual ~Button() = default;

    void SetClicked(GCallback callback, gpointer user_data)
    {
        g_signal_connect(this->button_, "clicked", callback, user_data);
    }

    GtkWidget *Get()
    {
        return this->button_;
    }

  private:
    GtkWidget *button_;
};

class DropDown
{
  public:
    DropDown()
    {
        this->string_list_ = gtk_string_list_new(nullptr);
        this->dropdown_ = gtk_drop_down_new(G_LIST_MODEL(this->string_list_), nullptr);
        // gtk_widget_set_hexpand(this->dropdown_, true);
        gtk_drop_down_set_factory(GTK_DROP_DOWN(this->dropdown_), CreateEllipsizeFactory());
        gtk_drop_down_set_list_factory(GTK_DROP_DOWN(this->dropdown_), CreateEllipsizeFactory());
    }

    ~DropDown() = default;

    void AppendText(const char *text)
    {
        gtk_string_list_append(this->string_list_, text);
        gtk_widget_add_css_class(this->dropdown_, "retinify_drop_down");
    }

    void RemoveAll()
    {
        gtk_string_list_splice(this->string_list_, 0, g_list_model_get_n_items(G_LIST_MODEL(this->string_list_)),
                               nullptr);
    }

    void SetActive(int index)
    {
        gtk_drop_down_set_selected(GTK_DROP_DOWN(this->dropdown_), index);
    }

    const char *GetActive()
    {
        GtkStringList *string_list = GTK_STRING_LIST(gtk_drop_down_get_model(GTK_DROP_DOWN(this->dropdown_)));
        int selected = gtk_drop_down_get_selected(GTK_DROP_DOWN(this->dropdown_));
        return gtk_string_list_get_string(string_list, selected);
    }

    GtkWidget *Get()
    {
        return this->dropdown_;
    }

  private:
    static GtkListItemFactory *CreateEllipsizeFactory()
    {
        auto *factory = gtk_signal_list_item_factory_new();
        g_signal_connect(factory, "setup", G_CALLBACK(SetupListItem), nullptr);
        g_signal_connect(factory, "bind", G_CALLBACK(BindListItem), nullptr);
        return factory;
    }

    static void SetupListItem(GtkListItemFactory *factory, GtkListItem *item)
    {
        auto *label = gtk_label_new(nullptr);
        gtk_label_set_ellipsize(GTK_LABEL(label), PANGO_ELLIPSIZE_MIDDLE);
        gtk_list_item_set_child(item, label);
    }

    static void BindListItem(GtkListItemFactory *factory, GtkListItem *item)
    {
        auto *label = GTK_LABEL(gtk_list_item_get_child(item));
        auto *string_object = GTK_STRING_OBJECT(gtk_list_item_get_item(item));
        gtk_label_set_text(label, gtk_string_object_get_string(string_object));
    }

    GtkWidget *dropdown_;
    GtkStringList *string_list_;
};

class Grid
{
  public:
    Grid()
    {
        this->grid_ = gtk_grid_new();
        // gtk_widget_set_hexpand(this->grid_, true);
        gtk_widget_add_css_class(this->grid_, "retinify_grid");
    }

    virtual ~Grid() = default;

    void Attach(GtkWidget *widget, int column, int row, int width, int height)
    {
        gtk_grid_attach(GTK_GRID(this->grid_), widget, column, row, width, height);
    }

    GtkWidget *Get()
    {
        return this->grid_;
    }

  private:
    GtkWidget *grid_;
};

class Expander
{
  public:
    Expander(const char *label)
    {
        this->expander_ = gtk_expander_new(label);
        this->content_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
        // gtk_widget_set_hexpand(this->content_, true);
        gtk_expander_set_child(GTK_EXPANDER(this->expander_), this->content_);
        gtk_widget_add_css_class(this->expander_, "retinify_expander");
    }

    virtual ~Expander() = default;

    void Append(GtkWidget *widget)
    {
        gtk_box_append(GTK_BOX(this->content_), widget);
    }

    void SetExpanded(bool expanded)
    {
        gtk_expander_set_expanded(GTK_EXPANDER(this->expander_), expanded);
    }

    GtkWidget *Get()
    {
        return this->expander_;
    }

  private:
    GtkWidget *expander_;
    GtkWidget *content_;
};

class ScrollWindow
{
  public:
    ScrollWindow()
    {
        this->scroll_ = gtk_scrolled_window_new();
        this->content_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
        gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(this->scroll_), this->content_);
        // gtk_widget_set_hexpand(this->scroll_, true);
        gtk_widget_set_vexpand(this->scroll_, true);
        gtk_widget_add_css_class(this->scroll_, "retinify_scroll_window");
    }

    virtual ~ScrollWindow() = default;

    void Append(GtkWidget *widget)
    {
        gtk_box_append(GTK_BOX(this->content_), widget);
    }

    GtkWidget *Get()
    {
        return this->scroll_;
    }

  private:
    GtkWidget *scroll_;
    GtkWidget *content_;
};

class List
{
  public:
    List()
    {
        this->list_ = gtk_list_box_new();
        // gtk_widget_set_hexpand(this->list_, true);
        gtk_widget_add_css_class(this->list_, "retinify_list");
    }

    ~List() = default;

    void Append(GtkWidget *widget)
    {
        gtk_list_box_append(GTK_LIST_BOX(this->list_), widget);
    }

    GtkWidget *Get()
    {
        return this->list_;
    }

  private:
    GtkWidget *list_;
};
} // namespace retinify