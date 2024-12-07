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
#include <opencv2/opencv.hpp>
#include <retinify/core.hpp>
namespace retinify
{
// variable
constexpr auto PAT_ROW = 4;
constexpr auto PAT_COL = 7;
constexpr auto CHESS_SIZE = 5;
// not variable
constexpr auto SUBPIX_WIN_SIZE = 11;
constexpr auto GAUSSIAN_SIGMA = 5;
constexpr auto CORNER_SUBPIX_WIN_SIZE = 5;
constexpr auto CORNER_SUBPIX_MAX_ITERATIONS = 40;
constexpr auto CORNER_SUBPIX_TERMINATION_EPSILON = 0.01;
constexpr auto STOCK_SIZE = 16;
constexpr auto ASPECT_MAX_MIN = 0.4;

class Calibration
{
  public:
    Calibration()
    {
        objp = GenerateObjectPoints();
    };
    ~Calibration() = default;

    inline std::vector<cv::Point3f> GenerateObjectPoints()
    {
        std::vector<cv::Point3f> objp(PAT_ROW * PAT_COL);
        for (int i = 0; i < PAT_ROW; ++i)
        {
            for (int j = 0; j < PAT_COL; ++j)
            {
                objp[i * PAT_COL + j] = cv::Point3f(j * CHESS_SIZE, i * CHESS_SIZE, 0);
            }
        }
        return objp;
    }

    inline void DefineRegionOfInterest(retinify::CalibrationData &calib_data)
    {
        float scale_max;
        if (this->state < 8)
        {
            scale_max = 1.0f;
        }
        else
        {
            scale_max = 0.8f;
        }

        float scale_min = scale_max * ASPECT_MAX_MIN;
        cv::Size imageSize = calib_data.GetInputImageSize();

        switch (this->state % 8)
        {
        case 0: // center
        case 2: // center
        case 4: // center
        case 6: // center
            this->roi_max = cv::Rect(imageSize.width * (1 - scale_max) / 2, imageSize.height * (1 - scale_max) / 2,
                                     imageSize.width * scale_max, imageSize.height * scale_max);
            break;
        case 1: // top-left
            this->roi_max = cv::Rect(0, 0, imageSize.width * scale_max / 2, imageSize.height * scale_max / 2);
            break;
        case 3: // top-right
            this->roi_max = cv::Rect(imageSize.width * (1 - scale_max / 2), 0, imageSize.width * scale_max / 2,
                                     imageSize.height * scale_max / 2);
            break;
        case 5: // bottom-left
            this->roi_max = cv::Rect(0, imageSize.height * (1 - scale_max / 2), imageSize.width * scale_max / 2,
                                     imageSize.height * scale_max / 2);
            break;
        case 7: // bottom-right
            this->roi_max = cv::Rect(imageSize.width * (1 - scale_max / 2), imageSize.height * (1 - scale_max / 2),
                                     imageSize.width * scale_max / 2, imageSize.height * scale_max / 2);
            break;
        }

        this->roi_min = cv::Rect(this->roi_max.x + this->roi_max.width * (1 - scale_min) / 2,
                                 this->roi_max.y + this->roi_max.height * (1 - scale_min) / 2,
                                 this->roi_max.width * scale_min, this->roi_max.height * scale_min);

        this->roi_quad_sub[0] =
            cv::Rect(this->roi_max.x, this->roi_max.y, this->roi_max.width, this->roi_max.height / 2);
        this->roi_quad_sub[1] = cv::Rect(this->roi_max.x, this->roi_max.y + this->roi_max.height / 2,
                                         this->roi_max.width, this->roi_max.height / 2);
        this->roi_quad_sub[2] =
            cv::Rect(this->roi_max.x, this->roi_max.y, this->roi_max.width / 2, this->roi_max.height);
        this->roi_quad_sub[3] = cv::Rect(this->roi_max.x + this->roi_max.width / 2, this->roi_max.y,
                                         this->roi_max.width / 2, this->roi_max.height);
    }

    inline bool IsPointsInRoi(const std::vector<cv::Point2f> &corners)
    {
        int min_num = 0;
        // constexpr int num_threshold = (PAT_COL * PAT_ROW) / 4 - 1;
        constexpr int num_threshold = (PAT_COL * PAT_ROW) / 8 - 1;
        std::array<int, 4> quad_count = {0, 0, 0, 0};

        for (const auto &point : corners)
        {
            if (!this->roi_max.contains(point))
                return false;
            if (this->roi_min.contains(point))
                min_num++;
            for (int i = 0; i < 4; ++i)
            {
                if (this->roi_quad_sub[i].contains(point))
                    quad_count[i]++;
            }
        }

        if (min_num >= corners.size() * 0.5)
        {
            return false;
        };

        return std::all_of(quad_count.begin(), quad_count.end(), [&](const int &num) { return num > num_threshold; });
    }

    inline void Compute(retinify::CalibrationData &calib_data)
    {
        cv::Size inputImageSize;
        std::array<cv::Mat, 2> cameraMatrix;
        std::array<cv::Mat, 2> distCoeffs;
        std::array<cv::Mat, 2> R;
        std::array<cv::Mat, 2> P;
        cv::Mat Q;
        std::array<cv::Rect, 2> valid;

        inputImageSize = calib_data.GetInputImageSize();
        cameraMatrix[0] = initCameraMatrix2D(object_points_stock, image_points_stock[0], inputImageSize, 0);
        cameraMatrix[1] = initCameraMatrix2D(object_points_stock, image_points_stock[1], inputImageSize, 0);

        cv::Mat R_raw, T_raw, E, F;
        double final_rms =
            cv::stereoCalibrate(object_points_stock, image_points_stock[0], image_points_stock[1], cameraMatrix[0],
                                distCoeffs[0], cameraMatrix[1], distCoeffs[1], inputImageSize, R_raw, T_raw, E, F,
                                cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_ZERO_TANGENT_DIST +
                                    cv::CALIB_USE_INTRINSIC_GUESS + cv::CALIB_SAME_FOCAL_LENGTH +
                                    cv::CALIB_RATIONAL_MODEL + cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5,
                                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));
        std::cout << "\x1b[32m" << "Final RMSE: " << final_rms << "\x1b[0m" << std::endl;

        cv::stereoRectify(cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], inputImageSize, R_raw, T_raw,
                          R[0], R[1], P[0], P[1], Q, cv::CALIB_ZERO_DISPARITY, 1, inputImageSize, &valid[0], &valid[1]);

        calib_data.SetInputImageSize(inputImageSize);
        calib_data.SetCameraMatrix(cameraMatrix);
        calib_data.SetDistCoeffs(distCoeffs);
        calib_data.SetR(R);
        calib_data.SetP(P);
        calib_data.SetQ(Q);
        calib_data.SetValid(valid);

        std::cout << "imageSize" << inputImageSize << std::endl;
        std::cout << "validRoi[0]" << valid[0] << std::endl;
        std::cout << "validRoi[1]" << valid[1] << std::endl;

        calib_data.Write(RETINIFY_DEFAULT_CALIBRATION_FILE_PATH);

        image_points_stock[0].clear();
        image_points_stock[1].clear();
        object_points_stock.clear();
    }

    //   private:
    cv::Rect roi_max, roi_min;
    std::array<cv::Rect, 4> roi_quad_sub;
    int state{0};
    std::vector<cv::Point3f> objp;
    std::vector<std::vector<cv::Point3f>> objectPoints, object_points_stock;
    std::vector<std::vector<cv::Point2f>> imagePoints[2], image_points_stock[2];
};
} // namespace retinify
