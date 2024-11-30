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
#include <GLES3/gl3.h>
#include <glviewer/glviewer.hpp>

// マウスドラッグ開始時の処理
inline static gboolean OnDragBegin(GtkGestureDrag *gesture, gdouble start_x, gdouble start_y, gpointer user_data)
{
    retinify::GLViewer *ctx = (retinify::GLViewer *)user_data;
    ctx->is_dragging = true;
    ctx->dragged_point = glm::normalize(ctx->camera_position);

    gtk_gl_area_queue_render(GTK_GL_AREA(ctx->gl_area));

    return G_SOURCE_CONTINUE;
}

// ドラッグ更新時の処理
inline static gboolean OnDragUpdate(GtkGestureDrag *gesture, gdouble offset_x, gdouble offset_y, gpointer user_data)
{
    retinify::GLViewer *ctx = (retinify::GLViewer *)user_data;
    if (!ctx->is_dragging) return G_SOURCE_CONTINUE;

    // 回転角度のスケーリング（感度調整）
    const float sensitivity = 0.005f; // 感度を調整
    const float delta_theta = offset_x * sensitivity;
    const float delta_phi = -offset_y * sensitivity;

    // 新しい角度を計算
    float theta = atan2(ctx->dragged_point.z, ctx->dragged_point.x);
    float phi = acos(ctx->dragged_point.y / glm::length(ctx->dragged_point));

    theta += delta_theta;
    phi = glm::clamp(phi + delta_phi, 0.1f, glm::pi<float>() - 0.1f);

    // 新しいカメラ位置を計算
    float radius = glm::length(ctx->camera_position);
    ctx->camera_position = glm::vec3(radius * sin(phi) * cos(theta), radius * cos(phi), radius * sin(phi) * sin(theta));

    // カメラの「前方向」ベクトルを計算
    glm::vec3 camera_forward = glm::normalize(-ctx->camera_position);

    // ワールド座標系の「上方向」ベクトル
    glm::vec3 world_up = glm::vec3(0.0f, 1.0f, 0.0f);

    // 「右方向」ベクトルを計算
    glm::vec3 camera_right = glm::normalize(glm::cross(world_up, camera_forward));

    // 新しい「上方向」ベクトルを計算
    ctx->up = glm::normalize(glm::cross(camera_forward, camera_right));

    // 描画を更新
    gtk_gl_area_queue_render(GTK_GL_AREA(ctx->gl_area));

    return G_SOURCE_CONTINUE;
}

// マウスドラッグ終了時の処理
inline static gboolean OnDragEnd(GtkGestureDrag *gesture, gdouble offset_x, gdouble offset_y, gpointer user_data)
{
    retinify::GLViewer *ctx = (retinify::GLViewer *)user_data;
    ctx->is_dragging = false;

    gtk_gl_area_queue_render(GTK_GL_AREA(ctx->gl_area));

    return G_SOURCE_CONTINUE;
}

// マウスホイール処理
inline static gboolean OnScroll(GtkEventControllerScroll *controller, gdouble dx, gdouble dy, gpointer user_data)
{
    retinify::GLViewer *ctx = (retinify::GLViewer *)user_data;
    ctx->zoom_level *= (1.0 + dy * 0.3);
    ctx->zoom_level = CLAMP(ctx->zoom_level, 0.01, 5.0);
    gtk_gl_area_queue_render(GTK_GL_AREA(ctx->gl_area));
    return G_SOURCE_CONTINUE;
}