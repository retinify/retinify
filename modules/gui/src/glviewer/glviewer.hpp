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
#include <gdk-pixbuf/gdk-pixbuf.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <gtk/gtk.h>
#include <math.h>
#include <retinify/pipeline.hpp>
#include <glviewer/buffer.hpp>
#include <format/format.hpp>
#define retinify_get_gui_glviewer retinify::GLViewer::Instance()
namespace retinify
{
class GLViewer : public Context<GLViewer>
{
  public:
    GLViewer();
    ~GLViewer();

    void AddOverlay(GtkWidget *widget);
    void ApplyConfiguration(const int width, const int height);
    void UpdatePCDPositionsAndColors(StereoImageData &data);

    GtkWidget *overlay{nullptr};
    GtkWidget *gl_viewer_box{nullptr};
    GtkWidget *gl_area{nullptr};

    PointCloudBuffer pcd_;

    // event handling
    bool is_dragging{false};
    glm::vec3 dragged_point{0.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    float zoom_level{1.0f};

    // buffers

    GLuint axis_position_buffer{0};
    GLuint axis_color_buffer{0};

    GLuint frame_position_buffer{0};
    GLuint frame_color_buffer{0};

    // camera matrices
    GLuint axis_vao{0};
    GLuint frame_vao{0};
    GLuint program{0};
    GLuint m_location{0};
    GLuint v_location{0};
    GLuint p_location{0};
    glm::mat4 view_matrix{0.0f};
    glm::mat4 projection_matrix{0.0f};
    glm::vec3 camera_position{0.0f, 0.0f, -10.0f};
};
} // namespace retinify