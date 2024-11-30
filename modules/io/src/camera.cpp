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

#include <retinify/camera.hpp>
#include <retinify/core.hpp>
class retinify::CameraThread::Impl
{
public:
    Impl()
    {
        l_grab_thread_ = std::make_unique<retinify::Thread>([this]() { this->Grab(this->l_cap_); });
        r_grab_thread_ = std::make_unique<retinify::Thread>([this]() { this->Grab(this->r_cap_); });
        retieve_thread_ = std::make_unique<retinify::Thread>([this]() { this->Retrieve(); });
    }

    ~Impl()
    {
        this->Stop();
    }

    inline void SetOutputQueue(retinify::Queue<retinify::StereoImageData> *queue)
    {
        this->queue_ = queue;
    }

    void Start(const char *node1, const char *node2)
    {
        this->Open(node1, node2);
        l_grab_thread_->Start();
        r_grab_thread_->Start();
        retieve_thread_->Start();
    }

    void Stop()
    {
        l_grab_thread_->Stop();
        r_grab_thread_->Stop();
        retieve_thread_->Stop();
        this->Close();
    }

private:
    void Open(const char *node1, const char *node2)
    {
        l_cap_.open(node1);
        r_cap_.open(node2);

        if (!l_cap_.isOpened())
        {
            throw std::runtime_error("Error: Unable to open the device at path '" + std::string(node1) + "'");
        }

        if (!r_cap_.isOpened())
        {
            throw std::runtime_error("Error: Unable to open the device at path '" + std::string(node2) + "'");
        }

        if (!l_cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')))
        {
            throw std::runtime_error("Error: Unable to set the codec");
        }

        if(!r_cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')))
        {
            throw std::runtime_error("Error: Unable to set the codec");
        }

        if (!(l_cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1280) && l_cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720)))
        {
            throw std::runtime_error("Error: Unable to set the frame size");
        }

        if (!(r_cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1280) && r_cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720)))
        {
            throw std::runtime_error("Error: Unable to set the frame size");
        }

        if (!l_cap_.set(cv::CAP_PROP_FPS, 30))
        {
            throw std::runtime_error("Error: Unable to set the frame rate");
        }

        if (!r_cap_.set(cv::CAP_PROP_FPS, 30))
        {
            throw std::runtime_error("Error: Unable to set the frame rate");
        }
    }

    void Grab(cv::VideoCapture &cap)
    {
        cap.grab();
    }

    void Retrieve()
    {
        retinify::StereoImageData data;
        if (l_cap_.retrieve(data.left_.image_) && r_cap_.retrieve(data.right_.image_))
        {
            queue_->Replace(std::make_unique<retinify::StereoImageData>(data));
        }
    }

    void Close()
    {
        l_cap_.release();
        r_cap_.release();
    }

    cv::VideoCapture l_cap_;
    cv::VideoCapture r_cap_;
    retinify::Queue<retinify::StereoImageData> *queue_;
    std::unique_ptr<retinify::Thread> l_grab_thread_;
    std::unique_ptr<retinify::Thread> r_grab_thread_;
    std::unique_ptr<retinify::Thread> retieve_thread_;
};

retinify::CameraThread::CameraThread()
{
    this->impl_ = std::make_unique<Impl>();
}

retinify::CameraThread::~CameraThread()
{
}

void retinify::CameraThread::Start(const char *node1, const char *node2)
{
    this->impl_->Start(node1, node2);
}

void retinify::CameraThread::Stop()
{
    this->impl_->Stop();
}

bool retinify::CameraThread::SetOutputQueue(retinify::Queue<retinify::StereoImageData> *queue)
{
    this->impl_->SetOutputQueue(queue);
    return true;
}