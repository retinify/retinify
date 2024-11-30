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

inline static void RenderPointCloud(retinify::GLViewer *ctx)
{
    // get window size
    int width = gtk_widget_get_width(GTK_WIDGET(ctx->gl_area));
    int height = gtk_widget_get_height(GTK_WIDGET(ctx->gl_area));
    float aspect_ratio = (float)width / (float)height;

    // OpenGL state
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // use shader program
    glUseProgram(ctx->program);

    // set up camera matrices
    glm::mat4 model_matrix(1.0f);
    ctx->view_matrix = glm::lookAt(ctx->camera_position * ctx->zoom_level, glm::vec3(0, 0, 0), ctx->up);
    ctx->projection_matrix = glm::perspective(glm::radians(30.0f), aspect_ratio, 0.01f, 100.0f);
    glUniformMatrix4fv(ctx->m_location, 1, GL_FALSE, glm::value_ptr(model_matrix));
    glUniformMatrix4fv(ctx->v_location, 1, GL_FALSE, glm::value_ptr(ctx->view_matrix));
    glUniformMatrix4fv(ctx->p_location, 1, GL_FALSE, glm::value_ptr(ctx->projection_matrix));

    ctx->pcd_.Draw();

    glBindVertexArray(ctx->axis_vao);
    glDrawArrays(GL_LINES, 0, 6);
    glLineWidth(2.0f);
    glBindVertexArray(0);

    glBindVertexArray(ctx->frame_vao);
    glDrawArrays(GL_LINES, 0, 16);
    glLineWidth(1.0f);
    glBindVertexArray(0);

    // error check
    GLenum error = glGetError();
    if (error != GL_NO_ERROR)
    {
        std::cerr << "OpenGL Error: " << error << std::endl;
    }
}
