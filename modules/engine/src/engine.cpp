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

#include <retinify/calibration.hpp>
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
            this->output_queue_->Replace(std::move(data));
        }
    }

    inline void Rectify()
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
        cv::Mat gray[2];
        auto data = this->GetImageData();

        calibration_->DefineRegionOfInterest(*this->calib_data_);

        cv::cvtColor(data->left_.image_, gray[0], cv::COLOR_BGR2GRAY);
        cv::cvtColor(data->right_.image_, gray[1], cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners[2];
        bool found1 = cv::findChessboardCorners(gray[0], cv::Size(PAT_COL, PAT_ROW), corners[0],
                                                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
                                                    cv::CALIB_CB_FAST_CHECK);

        bool found2 = cv::findChessboardCorners(gray[1], cv::Size(PAT_COL, PAT_ROW), corners[1],
                                                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
                                                    cv::CALIB_CB_FAST_CHECK);

        if (found1 && found2)
        {
            cv::cornerSubPix(gray[0], corners[0], cv::Size(CORNER_SUBPIX_WIN_SIZE, CORNER_SUBPIX_WIN_SIZE),
                             cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                              CORNER_SUBPIX_MAX_ITERATIONS, CORNER_SUBPIX_TERMINATION_EPSILON));

            cv::cornerSubPix(gray[1], corners[1], cv::Size(CORNER_SUBPIX_WIN_SIZE, CORNER_SUBPIX_WIN_SIZE),
                             cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                              CORNER_SUBPIX_MAX_ITERATIONS, CORNER_SUBPIX_TERMINATION_EPSILON));

            bool result1 = calibration_->IsPointsInRoi(corners[0]);
            bool result2 = calibration_->IsPointsInRoi(corners[1]);

            if (result1 && result2)
            {
                cv::find4QuadCornerSubpix(gray[0], corners[0], cv::Size(SUBPIX_WIN_SIZE, SUBPIX_WIN_SIZE));
                cv::find4QuadCornerSubpix(gray[1], corners[1], cv::Size(SUBPIX_WIN_SIZE, SUBPIX_WIN_SIZE));

                calibration_->image_points_stock[0].push_back(corners[0]);
                calibration_->image_points_stock[1].push_back(corners[1]);
                calibration_->object_points_stock.push_back(calibration_->objp);

                calibration_->state++;

                if (calibration_->image_points_stock[0].size() >= STOCK_SIZE &&
                    calibration_->image_points_stock[1].size() >= STOCK_SIZE)
                {
                    calibration_->Compute(*this->calib_data_);
                    return;
                }
            }
        }

        cv::cvtColor(gray[0], gray[0], cv::COLOR_GRAY2BGR);
        cv::cvtColor(gray[1], gray[1], cv::COLOR_GRAY2BGR);
        cv::rectangle(gray[0], calibration_->roi_max, cv::Scalar(255, 255, 0), 2);
        cv::rectangle(gray[1], calibration_->roi_max, cv::Scalar(255, 255, 0), 2);
        cv::rectangle(gray[0], calibration_->roi_min, cv::Scalar(255, 255, 0), 2);
        cv::rectangle(gray[1], calibration_->roi_min, cv::Scalar(255, 255, 0), 2);
        cv::drawChessboardCorners(gray[0], cv::Size(PAT_COL, PAT_ROW), corners[0], found1);
        cv::drawChessboardCorners(gray[1], cv::Size(PAT_COL, PAT_ROW), corners[1], found2);

        data->left_.image_ = gray[0];
        data->right_.image_ = gray[1];

        this->output_queue_->Replace(std::move(data));
    }

    inline void PostProcess(std::unique_ptr<retinify::StereoImageData> &data, bool gl = true)
    {
        float width_scale = static_cast<float>(data->left_.image_.cols) / data->disparity_.cols;
        cv::Mat scaled_disparity;
        data->disparity_.convertTo(scaled_disparity, CV_32F, width_scale);
        cv::resize(scaled_disparity, data->disparity_, data->left_.image_.size());

        // reproject disparity to 3D points
        cv::reprojectImageTo3D(data->disparity_, data->pcd_, this->calib_data_->GetQ(), true);
        if (gl)
        {
            float *ptr = data->pcd_.ptr<float>();
            for (int i = 0; i < data->pcd_.total(); ++i)
            {
                ptr[i * 3] = -ptr[i * 3];
                ptr[i * 3 + 1] = -ptr[i * 3 + 1];
            }
        }
    }

    inline void Inference()
    {
        auto data = this->GetImageData();
        if (data)
        {
            cv::remap(data->left_.image_, data->left_.image_, this->l_mapx_, this->l_mapy_, cv::INTER_LINEAR);
            cv::remap(data->right_.image_, data->right_.image_, this->r_mapx_, this->r_mapy_, cv::INTER_LINEAR);
            this->session_->UploadInputData(0, data->left_.image_);
            this->session_->UploadInputData(1, data->right_.image_);
            this->session_->Run();
            this->session_->DownloadOutputData(0, data->disparity_);
            this->PostProcess(data);
            this->output_queue_->Replace(std::move(data));
        }
    }

    inline void InitUndistortRectifyMap()
    {
        cv::initUndistortRectifyMap(this->calib_data_->GetCameraMatrix()[0], this->calib_data_->GetDistCoeffs()[0],
                                    this->calib_data_->GetR()[0], this->calib_data_->GetP()[0],
                                    this->calib_data_->GetInputImageSize(), CV_32FC1, this->l_mapx_, this->l_mapy_);
        cv::initUndistortRectifyMap(this->calib_data_->GetCameraMatrix()[1], this->calib_data_->GetDistCoeffs()[1],
                                    this->calib_data_->GetR()[1], this->calib_data_->GetP()[1],
                                    this->calib_data_->GetInputImageSize(), CV_32FC1, this->r_mapx_, this->r_mapy_);
    }

    inline void Start(CalibrationData &calib_data, StereoEngine::Mode mode)
    {
        this->calib_data_ = std::make_shared<retinify::CalibrationData>(calib_data);

        switch (mode)
        {
        case retinify::StereoEngine::Mode::RAWIMAGE:
            this->thread_ = std::make_unique<retinify::Thread>([this]() { this->RawImage(); });
            break;

        case retinify::StereoEngine::Mode::RECTIFY:
            this->InitUndistortRectifyMap();
            this->thread_ = std::make_unique<retinify::Thread>([this]() { this->Rectify(); });
            break;

        case retinify::StereoEngine::Mode::CALIBRATION:
            this->calibration_ = std::make_unique<retinify::Calibration>();
            this->thread_ = std::make_unique<retinify::Thread>([this]() { this->Calibration(); });
            break;

        case retinify::StereoEngine::Mode::INFERENCE:
            this->InitUndistortRectifyMap();
            this->session_ = std::make_unique<retinify::Session>(RETINIFY_DEFALUT_MODEL_PATH);
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
        if (this->calibration_)
        {
            this->calibration_.reset();
        }
    }

  private:
    retinify::Queue<retinify::StereoImageData> *input_queue_;
    retinify::Queue<retinify::StereoImageData> *output_queue_;
    cv::Mat l_mapx_, l_mapy_, r_mapx_, r_mapy_;
    std::shared_ptr<retinify::CalibrationData> calib_data_;
    std::unique_ptr<retinify::Thread> thread_;
    std::unique_ptr<retinify::Session> session_;
    std::unique_ptr<retinify::Calibration> calibration_;
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

void retinify::StereoEngine::Start(CalibrationData &calib_data, StereoEngine::Mode mode)
{
    this->impl_->Start(calib_data, mode);
}

void retinify::StereoEngine::Stop()
{
    this->impl_->Stop();
}