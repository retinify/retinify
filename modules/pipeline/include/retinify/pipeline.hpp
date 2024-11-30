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
#include <atomic>
#include <map>
#include <opencv2/opencv.hpp>
#include <optional>
#include <retinify/core.hpp>
#include <retinify/io.hpp>
#define retinify_get_core retinify::Core::Instance()
#define RETINIFY_PIPELINE_API
namespace retinify
{
/// @brief stereo pipeline
class Pipeline
{
public:
    Pipeline();
    ~Pipeline();
    /// @brief pipeline mode
    enum class Mode
    {
        RAWIMAGE,
        RECTIFY,
        CALIBRATION,
        INFERENCE,
        LOADER
    };
    /// @brief use USB cameras and capture stereo images
    RETINIFY_PIPELINE_API void Start(const retinify::CalibrationData &config, const Mode mode = Mode::INFERENCE);
    /// @brief control the pipeline
    RETINIFY_PIPELINE_API void Stop();
    /// @brief Data IO
    RETINIFY_PIPELINE_API std::unique_ptr<StereoImageData> GetOutputData();
    RETINIFY_PIPELINE_API std::unique_ptr<retinify::StereoImageData> TryGetOutputData();

    Pipeline(const Pipeline &) = delete;
    Pipeline &operator=(const Pipeline &) = delete;

    Pipeline(Pipeline &&) noexcept = default;
    Pipeline &operator=(Pipeline &&) noexcept = default;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
/// @brief Core class
class Core : public Singleton<Core>
{
public:
    Core();
    ~Core();

    bool IsCoreRunning();
    void ActivateCore();
    void DeactivateCore();
    std::unique_ptr<Pipeline> &GetPipelinePtr();

private:
    std::unique_ptr<Pipeline> pipeline_;
    std::atomic<bool> running_{true};
};
} // namespace retinify