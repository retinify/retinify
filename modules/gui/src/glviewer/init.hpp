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
#include <retinify/pipeline.hpp>

inline static void InitAxesBuffers(retinify::GLViewer *ctx)
{
    // 軸の頂点データを定義
    const GLfloat axis_vertices[] = {
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // X軸
        0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, // Y軸
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f  // Z軸
    };

    const GLfloat axis_colors[] = {
        1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // X軸
        0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, // Y軸
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f  // Z軸
    };

    // 軸の頂点バッファ
    glGenBuffers(1, &ctx->axis_position_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, ctx->axis_position_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(axis_vertices), axis_vertices, GL_DYNAMIC_DRAW);

    // 軸の色バッファ
    glGenBuffers(1, &ctx->axis_color_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, ctx->axis_color_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(axis_colors), axis_colors, GL_DYNAMIC_DRAW);

    // VAOの生成
    glGenVertexArrays(1, &ctx->axis_vao);
    glBindVertexArray(ctx->axis_vao);

    // 頂点属性の設定
    glBindBuffer(GL_ARRAY_BUFFER, ctx->axis_position_buffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, ctx->axis_color_buffer);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);

    // VAOのバインドを解除
    glBindVertexArray(0);
}

inline static void InitFrameBuffers(retinify::GLViewer *ctx)
{
    int width = 1280;
    int height = 720;

    GLfloat falf_width = width / 200;
    GLfloat half_height = height / 200;
    GLfloat depth = 10.0f;

    GLfloat floor_vertices[] = {
        // 床の頂点データを定義
        0.0f,         0.0f,        0.0f,         falf_width,   half_height, depth,        0.0f,         0.0f,
        0.0f,         -falf_width, half_height,  depth,        0.0f,        0.0f,         0.0f,         -falf_width,
        -half_height, depth,       0.0f,         0.0f,         0.0f,        falf_width,   -half_height, depth,
        falf_width,   half_height, depth,        -falf_width,  half_height, depth,        -falf_width,  half_height,
        depth,        -falf_width, -half_height, depth,        -falf_width, -half_height, depth,        falf_width,
        -half_height, depth,       falf_width,   -half_height, depth,       falf_width,   half_height,  depth};

    GLfloat floor_colors[] = {// 床の色データを定義
                              0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                              0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                              0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                              0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

    // 床の頂点バッファ
    glGenBuffers(1, &ctx->frame_position_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, ctx->frame_position_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(floor_vertices), floor_vertices, GL_DYNAMIC_DRAW);

    // 床の色バッファ
    glGenBuffers(1, &ctx->frame_color_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, ctx->frame_color_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(floor_colors), floor_colors, GL_DYNAMIC_DRAW);

    glGenVertexArrays(1, &ctx->frame_vao);
    glBindVertexArray(ctx->frame_vao);

    // 頂点属性の設定
    glBindBuffer(GL_ARRAY_BUFFER, ctx->frame_position_buffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, ctx->frame_color_buffer);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);

    // VAOのバインドを解除
    glBindVertexArray(0);
}