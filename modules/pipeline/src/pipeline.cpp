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

#include <calibration.hpp>
#include <iostream>
#include <retinify/core.hpp>
#include <retinify/pipeline.hpp>
#include <retinify/engine.hpp>
#include <retinify/io.hpp>
/**
 * Pipeline
 */
retinify::Pipeline::Pipeline()
{
    this->impl_ = std::make_unique<Impl>();
}

retinify::Pipeline::~Pipeline() = default;

class retinify::Pipeline::Impl
{
public:
    Impl()
    {
        camera_.SetOutputQueue(&this->camera_queue_);
        engine_.SetInputQueue(&this->camera_queue_);
        engine_.SetOutputQueue(&this->engine_queue_);
    }
    
    ~Impl()
    {
        this->Stop();
    }

    inline void Stop()
    {
        this->camera_.Stop();
        this->engine_.Stop();
    }

    inline std::unique_ptr<retinify::StereoImageData> GetOutputData()
    {
        return this->engine_queue_.Pop();
    }

    inline std::unique_ptr<retinify::StereoImageData> TryGetOutputData()
    {
        return this->engine_queue_.TryPop();
    }

    inline void Start(const retinify::CalibrationData &config, const retinify::Pipeline::Mode mode)
    {
        this->Stop(); // if already active, deactivate first
        std::optional<retinify::DeviceData> device1 = GetDeviceBySerialNumber(config.GetSerial()[0]);
        std::optional<retinify::DeviceData> device2 = GetDeviceBySerialNumber(config.GetSerial()[1]);
        std::cout << "Device 1: " << device1->node_ << std::endl;
        std::cout << "Device 2: " << device2->node_ << std::endl;
        this->camera_.Start(device1->node_.c_str(), device2->node_.c_str());

        switch (mode)
        {
        case retinify::Pipeline::Mode::RAWIMAGE:
            engine_.Start(retinify::StereoEngineThread::Mode::RAWIMAGE);
            break;

        case retinify::Pipeline::Mode::RECTIFY:
            break;

        case retinify::Pipeline::Mode::CALIBRATION:
            break;

        case retinify::Pipeline::Mode::INFERENCE:
            engine_.Start(retinify::StereoEngineThread::Mode::INFERENCE);
            break;
        }
    }

    /**
     * CALIBRATION
     *
     *
     */
    inline void Calibration()
    {
        Calib calib;
        retinify::CalibrationData config;
        while (true)
        {
            cv::Mat frame[2], gray[2];
            std::unique_ptr<retinify::StereoImageData> data;
            // data = this->stereo_.Pop();

            frame[0] = data->left_.image_;
            frame[1] = data->right_.image_;

            calib.DefineRegionOfInterest(config);

            if (frame[0].empty() || frame[1].empty())
            {
                std::cerr << "Error: Empty frame" << std::endl;
                break;
            }

            cv::cvtColor(frame[0], gray[0], cv::COLOR_BGR2GRAY);
            cv::cvtColor(frame[1], gray[1], cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(gray[0], gray[0], cv::Size(GAUSSIAN_SIGMA, GAUSSIAN_SIGMA), 0);
            cv::GaussianBlur(gray[1], gray[1], cv::Size(GAUSSIAN_SIGMA, GAUSSIAN_SIGMA), 0);

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

                bool result1 = calib.IsPointsInRoi(corners[0]);
                bool result2 = calib.IsPointsInRoi(corners[1]);

                if (result1 && result2)
                {
                    try
                    {
                        cv::find4QuadCornerSubpix(gray[0], corners[0], cv::Size(SUBPIX_WIN_SIZE, SUBPIX_WIN_SIZE));
                        cv::find4QuadCornerSubpix(gray[1], corners[1], cv::Size(SUBPIX_WIN_SIZE, SUBPIX_WIN_SIZE));

                        calib.image_points_stock[0].push_back(corners[0]);
                        calib.image_points_stock[1].push_back(corners[1]);
                        calib.object_points_stock.push_back(calib.objp);

                        calib.state++;

                        if (calib.image_points_stock[0].size() >= STOCK_SIZE &&
                            calib.image_points_stock[1].size() >= STOCK_SIZE)
                        {
                            // this->stereo_.CloseStereoCameras();
                            calib.Compute(config);
                            break;
                        }
                    }
                    catch (const cv::Exception &e)
                    {
                        std::cerr << "OpenCV Exception: " << e.what() << std::endl;
                    }
                }
            }
            cv::Mat show1, show2;
            show1 = cv::Mat(frame[0].size(), CV_8UC3, cv::Scalar(255, 255, 255));
            show2 = cv::Mat(frame[1].size(), CV_8UC3, cv::Scalar(255, 255, 255));

            cv::rectangle(show1, calib.roi_max, cv::Scalar(255, 255, 0), 2);
            cv::rectangle(show2, calib.roi_max, cv::Scalar(255, 255, 0), 2);
            cv::rectangle(show1, calib.roi_min, cv::Scalar(255, 255, 0), 2);
            cv::rectangle(show2, calib.roi_min, cv::Scalar(255, 255, 0), 2);
            cv::drawChessboardCorners(show1, cv::Size(PAT_COL, PAT_ROW), corners[0], found1);
            cv::drawChessboardCorners(show2, cv::Size(PAT_COL, PAT_ROW), corners[1], found2);

            data->left_.image_ = show1;
            data->right_.image_ = show2;

            this->engine_queue_.Replace(std::move(data));
        }
    }

    /**
     * INFERENCE PROCESS
     *
     *
     */
    inline void PostProcess(std::unique_ptr<retinify::StereoImageData> &data, bool gl = true)
    {
        // resize disparity
        // cv::Size new_size = this->GetConfiguration().GetInputImageSize();
        /// @todo implement
        cv::Size new_size;
        float width_scale = static_cast<float>(new_size.width) / data->disparity_.cols;
        cv::Mat scaled_disparity;
        data->disparity_.convertTo(scaled_disparity, CV_32F, width_scale);
        cv::resize(scaled_disparity, data->disparity_, new_size);

        // reproject disparity to 3D points
        // cv::reprojectImageTo3D(data->disparity_, data->pcd_, this->GetConfiguration().GetQ(), true);
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

    inline void Inference(retinify::CalibrationData config)
    {
        cv::Mat map1x, map1y, map2x, map2y;

        cv::initUndistortRectifyMap(config.GetCameraMatrix()[0], config.GetDistCoeffs()[0], config.GetR()[0],
                                    config.GetP()[0], config.GetInputImageSize(), CV_32FC1, map1x, map1y);
        cv::initUndistortRectifyMap(config.GetCameraMatrix()[1], config.GetDistCoeffs()[1], config.GetR()[1],
                                    config.GetP()[1], config.GetInputImageSize(), CV_32FC1, map2x, map2y);

        // initialize engine
        retinify::StereoEngineThread engine;

        // std::vector<int64> init_input_shape = engine.GetInitialInputShape();
        // cv::Size init_input_size(init_input_shape[3], init_input_shape[2]);

        // std::vector<int64> temporal_input_shape = engine.GetTemporalInputShape();
        // cv::Size temporal_input_size(temporal_input_shape[3], temporal_input_shape[2]);

        std::vector<float> input1;
        std::vector<float> input2;
        std::vector<float> init_disp, temporal_disp;

        while (true)
        {
            // auto data = this->stereo_.Pop();
            std::unique_ptr<retinify::StereoImageData> data;
            cv::remap(data->left_.image_, data->left_.image_, map1x, map1y, cv::INTER_LINEAR);
            cv::remap(data->right_.image_, data->right_.image_, map2x, map2y, cv::INTER_LINEAR);
            // input1 = retinify::CVMatToFloatVector(data->left_.image_, temporal_input_size);
            // input2 = retinify::CVMatToFloatVector(data->right_.image_, temporal_input_size);
            // engine.InitialInference(input1, input2, temporal_disp);
            // engine.TemporalInference(input1, input2, init_disp, temporal_disp);

            // data->disparity_ = cv::Mat(temporal_input_size, CV_32F, temporal_disp.data());
            this->PostProcess(data);
            this->engine_queue_.Replace(std::move(data));

            // init_disp = std::move(temporal_disp);
        }
    }

private:
    retinify::Queue<retinify::StereoImageData> camera_queue_;
    retinify::Queue<retinify::StereoImageData> engine_queue_;
    retinify::CameraThread camera_;
    retinify::StereoEngineThread engine_;
};

void retinify::Pipeline::Stop()
{
    this->impl_->Stop();
}

std::unique_ptr<retinify::StereoImageData> retinify::Pipeline::GetOutputData()
{
    return this->impl_->GetOutputData();
}

std::unique_ptr<retinify::StereoImageData> retinify::Pipeline::TryGetOutputData()
{
    return this->impl_->TryGetOutputData();
}

void retinify::Pipeline::Start(const retinify::CalibrationData &config, const Mode mode)
{
    this->impl_->Start(config, mode);
}