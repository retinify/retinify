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

    inline void Start(retinify::CalibrationData &calib, const retinify::Pipeline::Mode mode)
    {
        this->Stop(); // if already active, deactivate first
        
        std::optional<retinify::DeviceData> device1 = GetDeviceBySerialNumber(calib.GetSerial()[0]);
        std::optional<retinify::DeviceData> device2 = GetDeviceBySerialNumber(calib.GetSerial()[1]);

        this->camera_.Start(device1->node_.c_str(), device2->node_.c_str());

        switch (mode)
        {
        case retinify::Pipeline::Mode::RAWIMAGE:
            engine_.Start(calib, retinify::StereoEngine::Mode::RAWIMAGE);
            break;

        case retinify::Pipeline::Mode::RECTIFY:
            break;

        case retinify::Pipeline::Mode::CALIBRATION:
            break;

        case retinify::Pipeline::Mode::INFERENCE:
            engine_.Start(calib, retinify::StereoEngine::Mode::INFERENCE);
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

private:
    retinify::Queue<retinify::StereoImageData> camera_queue_;
    retinify::Queue<retinify::StereoImageData> engine_queue_;
    retinify::Camera camera_;
    retinify::StereoEngine engine_;
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

void retinify::Pipeline::Start(retinify::CalibrationData &calib, const Mode mode)
{
    this->impl_->Start(calib, mode);
}