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

#include <retinify/engine.hpp>
#include <retinify/session.hpp>

class retinify::StereoEngine::Impl
{
public:
    Impl()
    {
    }

    ~Impl()
    {
        this->Stop();
    }

    inline void SetInputQueue(retinify::Queue<retinify::StereoImageData> *queue)
    {
        this->input_queue_ = queue;
    }

    inline void SetOutputQueue(retinify::Queue<retinify::StereoImageData> *queue)
    {
        this->output_queue_ = queue;
    }

    inline std::unique_ptr<retinify::StereoImageData> GetImageData()
    {
        auto data = this->input_queue_->Pop(std::chrono::milliseconds(1000));
        if (data)
        {
            return data;
        }
        return nullptr;
    }

    inline void RawImage()
    {
        auto data = this->GetImageData();
        if (data)
        {
            cv::remap(data->left_.image_, data->left_.image_, this->l_mapx_, this->l_mapy_, cv::INTER_LINEAR);
            cv::remap(data->right_.image_, data->right_.image_, this->r_mapx_, this->r_mapy_, cv::INTER_LINEAR);
            this->output_queue_->Replace(std::move(data));
        }
    }

    inline void Calibration()
    {
    }

    inline void PostProcess(std::unique_ptr<retinify::StereoImageData> &data, bool gl = true)
    {
        float width_scale = static_cast<float>(data->left_.image_.cols) / data->disparity_.cols;
        cv::Mat scaled_disparity;
        data->disparity_.convertTo(scaled_disparity, CV_32F, width_scale);
        cv::resize(scaled_disparity, data->disparity_, data->left_.image_.size());

        // reproject disparity to 3D points
        // cv::reprojectImageTo3D(data->disparity_, data->pcd_, this->GetConfiguration().GetQ(), true);
        // if (gl)
        // {
        //     float *ptr = data->pcd_.ptr<float>();
        //     for (int i = 0; i < data->pcd_.total(); ++i)
        //     {
        //         ptr[i * 3] = -ptr[i * 3];
        //         ptr[i * 3 + 1] = -ptr[i * 3 + 1];
        //     }
        // }
    }

    inline void Inference()
    {
        auto data = this->GetImageData();
        if (data)
        {
            this->session_->UploadInputData(0, data->left_.image_);
            this->session_->UploadInputData(1, data->right_.image_);
            this->session_->Run();
            this->session_->DownloadOutputData(0, data->disparity_);
            this->PostProcess(data);
            this->output_queue_->Replace(std::move(data));
        }
    }

    inline void InitUndistortRectifyMap(CalibrationData &calib)
    {
        cv::initUndistortRectifyMap(calib.GetCameraMatrix()[0], calib.GetDistCoeffs()[0], calib.GetR()[0],
                                    calib.GetP()[0], calib.GetInputImageSize(), CV_32FC1, this->l_mapx_, this->l_mapy_);
        cv::initUndistortRectifyMap(calib.GetCameraMatrix()[1], calib.GetDistCoeffs()[1], calib.GetR()[1],
                                    calib.GetP()[1], calib.GetInputImageSize(), CV_32FC1, this->r_mapx_, this->r_mapy_);
    }

    inline void Start(CalibrationData &calib, StereoEngine::Mode mode)
    {
        this->InitUndistortRectifyMap(calib);

        switch (mode)
        {
        case retinify::StereoEngine::Mode::RAWIMAGE:
            this->thread_ = std::make_unique<retinify::Thread>([this]() { this->RawImage(); });
            break;

        case retinify::StereoEngine::Mode::CALIBRATION:
            this->thread_ = std::make_unique<retinify::Thread>([this]() { this->Calibration(); });
            break;

        case retinify::StereoEngine::Mode::INFERENCE:
            this->session_ = std::make_unique<retinify::Session>("/model.onnx");
            this->thread_ = std::make_unique<retinify::Thread>([this]() { this->Inference(); });
            break;
        }

        this->thread_->Start();
    }

    inline void Stop()
    {
        if (this->thread_)
        {
            this->thread_->Stop();
            this->thread_.reset();
        }
        if (this->session_)
        {
            this->session_.reset();
        }
    }

private:
    cv::Mat l_mapx_, l_mapy_, r_mapx_, r_mapy_;
    retinify::Queue<retinify::StereoImageData> *input_queue_;
    retinify::Queue<retinify::StereoImageData> *output_queue_;
    std::unique_ptr<retinify::Thread> thread_;
    std::unique_ptr<retinify::Session> session_;
};

retinify::StereoEngine::StereoEngine()
{
    this->impl_ = std::make_unique<retinify::StereoEngine::Impl>();
}

retinify::StereoEngine::~StereoEngine() = default;

void retinify::StereoEngine::SetInputQueue(retinify::Queue<retinify::StereoImageData> *queue)
{
    this->impl_->SetInputQueue(queue);
}

void retinify::StereoEngine::SetOutputQueue(retinify::Queue<retinify::StereoImageData> *queue)
{
    this->impl_->SetOutputQueue(queue);
}

void retinify::StereoEngine::Start(CalibrationData &calib, StereoEngine::Mode mode)
{
    this->impl_->Start(calib, mode);
}

void retinify::StereoEngine::Stop()
{
    this->impl_->Stop();
}