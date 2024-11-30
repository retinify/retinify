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

inline static GLuint CreateShader(GLenum type, const char *src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint log_length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
        std::vector<char> log(log_length);
        glGetShaderInfoLog(shader, log_length, NULL, log.data());
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

inline static void InitShaders(retinify::GLViewer *ctx)
{
    const char *vertex_shader_source = "#version 320 es\n"
                                       "precision mediump float;\n"
                                       "uniform mat4 projection_matrix;\n"
                                       "uniform mat4 model_matrix;\n"
                                       "uniform mat4 view_matrix;\n"
                                       "layout(location=0) in vec3 in_position;\n"
                                       "layout(location=1) in vec3 in_color;\n"
                                       "out vec3 frag_color;\n"
                                       "void main(void) {\n"
                                       "    mat4 mv_matrix = view_matrix * model_matrix;\n"
                                       "    vec4 position_cameraspace = mv_matrix * vec4(in_position, 1);\n"
                                       "    gl_Position = projection_matrix * position_cameraspace;\n"
                                       "    frag_color = in_color;\n"
                                       "}\n";

    const char *fragment_shader_source = "#version 320 es\n"
                                         "precision mediump float;\n"
                                         "in vec3 frag_color;\n"
                                         "out vec4 out_frag_color;\n"
                                         "void main(void) {\n"
                                         "    out_frag_color = vec4(frag_color, 1.0);\n"
                                         "}\n";

    GLuint vertex = CreateShader(GL_VERTEX_SHADER, vertex_shader_source);
    GLuint fragment = CreateShader(GL_FRAGMENT_SHADER, fragment_shader_source);

    // シェーダーのエラーチェック
    if (vertex == 0 || fragment == 0)
    {
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        ctx->program = 0;
        ctx->m_location = 0;
        ctx->v_location = 0;
        ctx->p_location = 0;
        return;
    }

    // プログラムのリンク
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);

    // リンクのエラーチェック
    GLint link_status;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);
    if (link_status == GL_FALSE)
    {
        GLint log_len;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_len);
        if (log_len > 0)
        {
            std::vector<char> buffer(log_len);
            glGetProgramInfoLog(program, log_len, NULL, buffer.data());
            g_warning("Linking failure:\n%s", buffer.data());
        }
        glDeleteProgram(program);
        program = 0;
    }

    // uniform locationの取得
    ctx->m_location = glGetUniformLocation(program, "model_matrix");
    ctx->v_location = glGetUniformLocation(program, "view_matrix");
    ctx->p_location = glGetUniformLocation(program, "projection_matrix");

    glDetachShader(program, vertex);
    glDetachShader(program, fragment);
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    ctx->program = program;
}
