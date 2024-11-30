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
#include <retinify/pipeline.hpp>
#define GL_VIEWER_SCALE 0.5
namespace retinify
{
class PointCloudBuffer
{
  public:
    PointCloudBuffer() = default;
    ~PointCloudBuffer()
    {
        ReleaseResources();
    }

    void Init(int width, int height)
    {
        this->ReleaseResources();

        this->width = width;
        this->height = height;
        this->num = width * height;
        this->positions.resize(num * 3, 0.0f);
        this->colors.resize(num * 3, 0.5f);

        GLfloat aspect_ratio = static_cast<GLfloat>(width) / height;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = y * width + x;
                this->positions[index * 3] = -((static_cast<GLfloat>(x) / (width - 1)) * 2.0f - 1.0f) * 0.5f;
                this->positions[index * 3 + 1] =
                    -(((static_cast<GLfloat>(y) / (height - 1)) * 2.0f - 1.0f) / aspect_ratio) * 0.5f;
                this->positions[index * 3 + 2] = 0;
            }
        }

        glGenBuffers(1, &this->position_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, this->position_buffer);
        glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(GLfloat), positions.data(), GL_DYNAMIC_DRAW);

        glGenBuffers(1, &this->color_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, this->color_buffer);
        glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(GLfloat), colors.data(), GL_DYNAMIC_DRAW);

        glGenVertexArrays(1, &this->vao);
        glBindVertexArray(this->vao);

        glBindBuffer(GL_ARRAY_BUFFER, this->position_buffer);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, this->color_buffer);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
    }

    void Update(const void *pptr, int psize, const void *cptr, int csize)
    {
        // if (psize != this->positions.size() * sizeof(GLfloat) || csize != this->colors.size() * sizeof(GLfloat))
        // {
        //     std::cout << this->positions.size() * sizeof(GLfloat) << std::endl;
        //     std::cerr << "Invalid buffer size" << std::endl;
        // }

        std::memcpy(this->positions.data(), pptr, psize);
        std::memcpy(this->colors.data(), cptr, csize);

        glBindBuffer(GL_ARRAY_BUFFER, this->position_buffer);
        glBufferSubData(GL_ARRAY_BUFFER, 0, psize, positions.data());

        glBindBuffer(GL_ARRAY_BUFFER, this->color_buffer);
        glBufferSubData(GL_ARRAY_BUFFER, 0, csize, colors.data());
    }

    void Draw() const
    {
        glBindVertexArray(this->vao);
        glDrawArrays(GL_POINTS, 0, this->num);
        glBindVertexArray(0);
    }

  private:
    void ReleaseResources()
    {
        if (position_buffer)
            glDeleteBuffers(1, &position_buffer);
        if (color_buffer)
            glDeleteBuffers(1, &color_buffer);
        if (vao)
            glDeleteVertexArrays(1, &vao);

        position_buffer = 0;
        color_buffer = 0;
        vao = 0;
    }

    int num{0};
    int width{0};
    int height{0};
    std::vector<GLfloat> positions;
    std::vector<GLfloat> colors;
    GLuint position_buffer{0};
    GLuint color_buffer{0};
    GLuint vao{0};
};
} // namespace retinify