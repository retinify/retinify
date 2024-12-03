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
class retinify::Camera::Impl
{
public:
    Impl()
    {
        l_thread_ = std::make_unique<retinify::Thread>(
            [this]() { this->VideoCapture(this->l_cap_, this->l_frame_, this->l_ready_); });
        r_thread_ = std::make_unique<retinify::Thread>(
            [this]() { this->VideoCapture(this->r_cap_, this->r_frame_, this->r_ready_); });
        process_thread_ = std::make_unique<retinify::Thread>([this]() { this->Process(); });
    }

    ~Impl()
    {
        this->Stop();
    }

    void SetOutputQueue(retinify::Queue<retinify::StereoImageData> *queue)
    {
        this->queue_ = queue;
    }

    void Start(const char *node1, const char *node2)
    {
        this->Open(node1, node2);
        l_thread_->Start();
        r_thread_->Start();
        process_thread_->Start();
    }

    void Stop()
    {
        l_thread_->Stop();
        r_thread_->Stop();
        process_thread_->Stop();
        this->Close();
    }

private:
    void Open(const char *node1, const char *node2)
    {
        l_cap_.open(node1, cv::CAP_V4L2);
        r_cap_.open(node2, cv::CAP_V4L2);

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

        if (!r_cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')))
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

        if (!l_cap_.set(cv::CAP_PROP_BUFFERSIZE, 3))
        {
            throw std::runtime_error("Error: Unable to set the buffer size");
        }

        if (!r_cap_.set(cv::CAP_PROP_BUFFERSIZE, 3))
        {
            throw std::runtime_error("Error: Unable to set the buffer size");
        }
    }

    void VideoCapture(cv::VideoCapture &cap, retinify::ImageData &frame, bool &ready)
    {
        cv::Mat temp;
        cap.grab();
        cap.retrieve(temp);

        if (temp.empty())
        {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mtx);
            frame.image_ = temp.clone();
            ready = true;
        }

        cond.notify_one();
    }

    void Process()
    {
        retinify::StereoImageData data;
        std::unique_lock<std::mutex> lock(mtx);
        cond.wait_for(lock, std::chrono::milliseconds(1000), [this] { return l_ready_ && r_ready_; });

        if (!l_frame_.image_.empty() && !r_frame_.image_.empty())
        {
            data.left_ = l_frame_;
            data.right_ = r_frame_;
            queue_->Replace(std::make_unique<retinify::StereoImageData>(data));
            l_ready_ = false;
            r_ready_ = false;
        }
    }

    void Close()
    {
        l_cap_.release();
        r_cap_.release();
    }

    cv::VideoCapture l_cap_;
    cv::VideoCapture r_cap_;

    retinify::ImageData l_frame_;
    retinify::ImageData r_frame_;    

    std::mutex mtx;
    std::condition_variable cond;
    bool l_ready_{false};
    bool r_ready_{false};

    retinify::Queue<retinify::StereoImageData> *queue_;

    std::unique_ptr<retinify::Thread> l_thread_;
    std::unique_ptr<retinify::Thread> r_thread_;
    std::unique_ptr<retinify::Thread> process_thread_;
};

retinify::Camera::Camera()
{
    this->impl_ = std::make_unique<Impl>();
}

retinify::Camera::~Camera()
{
}

void retinify::Camera::Start(const char *node1, const char *node2)
{
    this->impl_->Start(node1, node2);
}

void retinify::Camera::Stop()
{
    this->impl_->Stop();
}

bool retinify::Camera::SetOutputQueue(retinify::Queue<retinify::StereoImageData> *queue)
{
    this->impl_->SetOutputQueue(queue);
    return true;
}