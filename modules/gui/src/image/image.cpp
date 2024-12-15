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

#include <glviewer/glviewer.hpp>
#include <image/image.hpp>
#include <retinify/core.hpp>

// create_texture_from_mat
GdkTexture *CreateTextureFromMat(const cv::Mat &image)
{
    GBytes *bytes = g_bytes_new(image.data, image.total() * image.elemSize());
    GdkTexture *texture = gdk_memory_texture_new(image.cols, image.rows, GDK_MEMORY_R8G8B8, bytes, image.step);
    g_bytes_unref(bytes);
    return texture;
}

retinify::ImageBox::ImageBox()
{
    // Create Widgets
    this->image_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 1);
    this->left_image_widget = gtk_picture_new();
    this->right_image_widget = gtk_picture_new();
    this->image3 = gtk_picture_new();
    this->image4 = gtk_picture_new();

    // init
    cv::Mat init_frame(this->image_size, CV_8UC3, cv::Scalar(0, 0, 0));
    auto texture = CreateTextureFromMat(init_frame);

    // set picture
    gtk_picture_set_paintable(GTK_PICTURE(this->left_image_widget), GDK_PAINTABLE(texture));
    gtk_picture_set_paintable(GTK_PICTURE(this->right_image_widget), GDK_PAINTABLE(texture));
    gtk_picture_set_paintable(GTK_PICTURE(this->image3), GDK_PAINTABLE(texture));
    gtk_picture_set_paintable(GTK_PICTURE(this->image4), GDK_PAINTABLE(texture));
    g_object_unref(texture);

    // image box
    gtk_widget_set_halign(this->image_box, GTK_ALIGN_CENTER);
    gtk_widget_set_hexpand(this->left_image_widget, TRUE);
    gtk_widget_set_hexpand(this->right_image_widget, TRUE);
    gtk_widget_set_hexpand(this->image3, TRUE);
    gtk_widget_set_hexpand(this->image4, TRUE);
    gtk_box_append(GTK_BOX(this->image_box), this->left_image_widget);
    gtk_box_append(GTK_BOX(this->image_box), this->right_image_widget);
    gtk_box_append(GTK_BOX(this->image_box), this->image3);
    gtk_box_append(GTK_BOX(this->image_box), this->image4);

    this->Append(this->image_box);
}

void retinify::ImageBox::UpdateDisplayStereoData(StereoImageData &data)
{
    try
    {
        cv::Mat show_left_image, show_right_image, show_disparity_image;
        cv::resize(data.left_.image_, show_left_image, this->image_size);
        cv::resize(data.right_.image_, show_right_image, this->image_size);
        show_disparity_image = retinify::ColoringDisparity(data.disparity_, 256, true);
        cv::resize(show_disparity_image, show_disparity_image, this->image_size);

        // Convert Mat to GdkTexture
        GdkTexture *left_texture = CreateTextureFromMat(show_left_image);
        GdkTexture *right_texture = CreateTextureFromMat(show_right_image);
        GdkTexture *disparity_texture = CreateTextureFromMat(show_disparity_image);

        if (!left_texture || !right_texture || !disparity_texture)
        {
            g_warning("Failed to create texture from image");
            return;
        }

        // Set paintable
        gtk_picture_set_paintable(GTK_PICTURE(this->left_image_widget), GDK_PAINTABLE(left_texture));
        gtk_picture_set_paintable(GTK_PICTURE(this->right_image_widget), GDK_PAINTABLE(right_texture));
        gtk_picture_set_paintable(GTK_PICTURE(this->image3), GDK_PAINTABLE(disparity_texture));

        // Cleanup
        // g_object_unref(left_texture);
        // g_object_unref(right_texture);
        // g_object_unref(disparity_texture);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        g_warning("Failed to update display stereo image");
    }
}