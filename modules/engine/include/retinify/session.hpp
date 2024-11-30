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
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>
namespace retinify
{
class Session
{
public:
    Session(std::string model_path);
    ~Session();
    std::vector<int64_t> GetInputShape(int index);
    std::vector<int64_t> GetOutputShape(int index);
    void UploadInputData(int index, const std::vector<float> &data);
    void UploadInputData(int index, const cv::Mat &data);
    void DownloadOutputData(int index, std::vector<float> &data);
    void DownloadOutputData(int index, cv::Mat &data);
    void DownloadOutputData(int index, cv::Mat &data, int type);
    void Run();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace retinify