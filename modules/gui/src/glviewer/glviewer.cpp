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
#include <glviewer/events.hpp>
#include <glviewer/glviewer.hpp>
#include <glviewer/init.hpp>
#include <glviewer/render.hpp>
#include <glviewer/shader.hpp>

// GLの状態を初期化する
inline static void Realize(GtkWidget *widget, gpointer user_data)
{
    retinify::GLViewer *ctx = (retinify::GLViewer *)user_data;
    gtk_gl_area_make_current(GTK_GL_AREA(widget));

    if (gtk_gl_area_get_error(GTK_GL_AREA(widget)) != NULL)
        return;

    // GLリソースの初期化
    ctx->pcd_.Init(1280, 720);
    InitAxesBuffers(ctx);
    InitFrameBuffers(ctx);
    InitShaders(ctx);
}

// GLの状態を破棄する
inline static gboolean Unrealize(GtkWidget *widget, gpointer user_data)
{
    retinify::GLViewer *ctx = (retinify::GLViewer *)user_data;
    gtk_gl_area_make_current(GTK_GL_AREA(widget));

    if (gtk_gl_area_get_error(GTK_GL_AREA(widget)) != NULL)
        return G_SOURCE_REMOVE;

    // GLリソースの解放
    glDeleteBuffers(1, &ctx->axis_position_buffer);
    glDeleteBuffers(1, &ctx->axis_color_buffer);
    glDeleteBuffers(1, &ctx->frame_position_buffer);
    glDeleteBuffers(1, &ctx->frame_color_buffer);
    glDeleteProgram(ctx->program);

    return G_SOURCE_CONTINUE;
}

// 描画コールバック
inline static gboolean Render(GtkWidget *widget, GdkGLContext *context, gpointer user_data)
{
    retinify::GLViewer *ctx = (retinify::GLViewer *)user_data;

    if (gtk_gl_area_get_error(GTK_GL_AREA(ctx->gl_area)) != NULL)
        return G_SOURCE_REMOVE;

    RenderPointCloud(ctx);

    return G_SOURCE_CONTINUE;
}

retinify::GLViewer::GLViewer()
{
    // create widgets
    this->overlay = gtk_overlay_new();
    this->gl_viewer_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    this->gl_area = gtk_gl_area_new();
    
    // gtk_gl_area_set_use_es(GTK_GL_AREA(this->gl_area), TRUE);
    gtk_gl_area_set_has_depth_buffer(GTK_GL_AREA(this->gl_area), TRUE);
    gtk_gl_area_set_auto_render(GTK_GL_AREA(this->gl_area), TRUE);

    // add widgets
    gtk_overlay_set_child(GTK_OVERLAY(overlay), this->gl_viewer_box);
    gtk_widget_set_hexpand(this->gl_area, TRUE);
    gtk_widget_set_vexpand(this->gl_area, TRUE);
    gtk_box_append(GTK_BOX(this->gl_viewer_box), this->gl_area);
    gtk_widget_add_css_class(this->gl_viewer_box, "gl-viewer-box");
    gtk_widget_add_css_class(this->overlay, "gl-viewer-overlay");
    gtk_widget_add_css_class(this->gl_area, "gl-area");

    // mouse events
    GtkGesture *drag_gesture = gtk_gesture_drag_new();
    GtkEventController *scroll_controller = gtk_event_controller_scroll_new(GTK_EVENT_CONTROLLER_SCROLL_BOTH_AXES);
    gtk_widget_add_controller(this->gl_area, GTK_EVENT_CONTROLLER(drag_gesture));
    gtk_widget_add_controller(this->gl_area, scroll_controller);

    this->Append(this->overlay);

    // connect signals
    g_signal_connect(this->gl_area, "realize", G_CALLBACK(Realize), this);
    g_signal_connect(this->gl_area, "unrealize", G_CALLBACK(Unrealize), this);
    g_signal_connect(this->gl_area, "render", G_CALLBACK(Render), this);
    g_signal_connect(drag_gesture, "drag-begin", G_CALLBACK(OnDragBegin), this);
    g_signal_connect(drag_gesture, "drag-update", G_CALLBACK(OnDragUpdate), this);
    g_signal_connect(drag_gesture, "drag-end", G_CALLBACK(OnDragEnd), this);
    g_signal_connect(scroll_controller, "scroll", G_CALLBACK(OnScroll), this);
}

retinify::GLViewer::~GLViewer() = default;

void retinify::GLViewer::AddOverlay(GtkWidget *widget)
{
    gtk_overlay_add_overlay(GTK_OVERLAY(this->overlay), widget);
}

void retinify::GLViewer::ApplyConfiguration(const int width, const int height)
{
    this->pcd_.Init(width, height);
    gtk_gl_area_queue_render(GTK_GL_AREA(this->gl_area));
}

void retinify::GLViewer::UpdatePCDPositionsAndColors(StereoImageData &data)
{
    cv::Mat points3D;
    data.pcd_.convertTo(points3D, CV_32FC3, 1.0 / 100.0);
    cv::Mat tmp;
    data.left_.image_.convertTo(tmp, CV_32FC3, 1.0 / 255.0);

    this->pcd_.Update(points3D.data, points3D.total() * points3D.elemSize(), tmp.data, tmp.total() * tmp.elemSize());
    gtk_gl_area_queue_render(GTK_GL_AREA(this->gl_area));
}