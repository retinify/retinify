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

#if USE_GPU
#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>
#endif
#include <iostream>
#include <numeric>
#include <retinify/session.hpp>
#include <vector>
#define TRT_ENGINE_CACHE_PATH "./trt_engines"
#define TRT_TIMING_CACHE_PATH "./trt_timing"
namespace retinify
{
#if USE_GPU
class Session::Impl
{
  public:
    inline Impl(std::string model_path)
    {
        // ONNX Runtime
        const auto &api = Ort::GetApi();

        // env
        this->env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING);

        // TensorRT Provider Options
        OrtTensorRTProviderOptionsV2 *tensorrt_options;
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));

        std::vector<const char *> option_keys = {
            "device_id",
            "trt_fp16_enable",
            "trt_engine_cache_enable",
            "trt_engine_cache_path",
            "trt_timing_cache_enable",
            "trt_timing_cache_path",
        };

        std::vector<const char *> option_values = {
            "0", "1", "1", TRT_ENGINE_CACHE_PATH, "1", TRT_TIMING_CACHE_PATH,
        };

        Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(tensorrt_options, option_keys.data(), option_values.data(),
                                                            option_keys.size()));

        Ort::SessionOptions session_options;
        // Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(session_options, tensorrt_options));

        OrtCUDAProviderOptions cuda_options;
        Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA(session_options, &cuda_options));

        this->session_ = Ort::Session(this->env_, ORT_TSTR(model_path.c_str()), session_options);

        this->allocator =
            Ort::Allocator(this->session_, Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
        this->info_cuda = Ort::MemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
        this->cuda_allocator = Ort::Allocator(this->session_, info_cuda);

        this->run_option_ = Ort::RunOptions();
        this->binding_ = Ort::IoBinding(this->session_);

        for (size_t i = 0; i < this->session_.GetInputCount(); i++)
        {
            Ort::TypeInfo type_info = this->session_.GetInputTypeInfo(i);
            std::vector<int64_t> shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
            size_t size = type_info.GetTensorTypeAndShapeInfo().GetElementCount();
            void *data = cuda_allocator.Alloc(size * sizeof(float));
            Ort::Value ort_value =
                Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float *>(data), size, shape.data(), shape.size());
            this->inputs_.push_back(IO{shape, size, data});
            Ort::AllocatedStringPtr input_name = this->session_.GetInputNameAllocated(i, this->allocator);
            this->binding_.BindInput(input_name.get(), ort_value);
        }

        for (size_t i = 0; i < this->session_.GetOutputCount(); i++)
        {
            Ort::TypeInfo type_info = this->session_.GetOutputTypeInfo(i);
            std::vector<int64_t> shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
            size_t size = type_info.GetTensorTypeAndShapeInfo().GetElementCount();
            void *data = cuda_allocator.Alloc(size * sizeof(float));
            Ort::Value ort_value =
                Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float *>(data), size, shape.data(), shape.size());
            this->outputs_.push_back(IO{shape, size, data});
            Ort::AllocatedStringPtr output_name = this->session_.GetOutputNameAllocated(i, this->allocator);
            this->binding_.BindOutput(output_name.get(), ort_value);
        }
    };

    inline ~Impl() = default;

    inline std::vector<int64_t> GetInputShape(int index) const
    {
        return this->inputs_[index].shape_;
    }

    inline std::size_t GetInputSize(int index) const
    {
        return this->inputs_[index].size_;
    }

    inline std::vector<int64_t> GetOutputShape(int index) const
    {
        return this->outputs_[index].shape_;
    }

    inline std::size_t GetOutputSize(int index) const
    {
        return this->outputs_[index].size_;
    }

    inline void UploadInputData(int index, const std::vector<float> &data)
    {
        // check if the index is valid
        if (index >= this->inputs_.size())
        {
            throw std::runtime_error("Invalid input index");
        }

        // check if the size which data ptr points to is valid
        if (data.size() != this->inputs_[index].size_)
        {
            throw std::runtime_error("Invalid input size");
        }

        cudaMemcpy(this->inputs_[index].data_, data.data(), sizeof(float) * this->inputs_[index].size_,
                   cudaMemcpyHostToDevice);
    }

    inline void DownloadOutputData(int index, std::vector<float> &data)
    {
        // check if the index is valid
        if (index >= this->outputs_.size())
        {
            throw std::runtime_error("Invalid output index");
        }

        // resize the output vector
        data.resize(this->outputs_[index].size_);

        cudaMemcpy(data.data(), this->outputs_[index].data_, sizeof(float) * this->outputs_[index].size_,
                   cudaMemcpyDeviceToHost);
    }

    inline void Run()
    {
        this->session_.Run(this->run_option_, this->binding_);
    }

  protected:
    Ort::Session session_{nullptr};
    Ort::Env env_{nullptr};
    Ort::IoBinding binding_{nullptr};
    Ort::RunOptions run_option_{nullptr};
    Ort::MemoryInfo info_cuda{nullptr};
    Ort::Allocator allocator{nullptr};
    Ort::Allocator cuda_allocator{nullptr};

    struct IO
    {
        std::vector<int64_t> shape_;
        size_t size_;
        void *data_{nullptr};
    };
    std::vector<IO> inputs_;
    std::vector<IO> outputs_;
};

Session::Session(std::string model_path)
{
    this->impl_ = std::make_unique<Impl>(model_path);
}

Session::~Session() = default;

std::vector<int64_t> Session::GetInputShape(int index)
{
    return this->impl_->GetInputShape(index);
}

std::vector<int64_t> Session::GetOutputShape(int index)
{
    return this->impl_->GetOutputShape(index);
}

void Session::UploadInputData(int index, const std::vector<float> &data)
{
    this->impl_->UploadInputData(index, data);
}

void Session::UploadInputData(int index, const cv::Mat &data)
{
    std::vector<int64_t> shape = this->impl_->GetInputShape(index);
    cv::Size size(shape[3], shape[2]);
    cv::Mat blob = cv::dnn::blobFromImage(data, 1.0, size, cv::Scalar(), true);
    this->impl_->UploadInputData(index, std::vector<float>(blob.begin<float>(), blob.end<float>()));
}

void Session::DownloadOutputData(int index, std::vector<float> &data)
{
    this->impl_->DownloadOutputData(index, data);
}

void Session::DownloadOutputData(int index, cv::Mat &data)
{
    std::vector<int64_t> shape = this->impl_->GetOutputShape(index);
    cv::Size size(shape[2], shape[1]);
    std::vector<float> output;
    this->DownloadOutputData(index, output);
    data = cv::Mat(size, CV_32F, output.data()).clone();
}

void Session::DownloadOutputData(int index, cv::Mat &data, int type)
{
    std::vector<int64_t> shape = this->impl_->GetOutputShape(index);
    cv::Size size(shape[2], shape[1]);
    std::vector<float> output;
    this->DownloadOutputData(index, output);
    data = cv::Mat(size, type, output.data()).clone();
}

void Session::Run()
{
    this->impl_->Run();
}
#else
class retinify::Session::Impl
{
};
std::vector<int64_t> retinify::Session::GetInputShape(int index)
{
    return {};
};
std::vector<int64_t> retinify::Session::GetOutputShape(int index)
{
    return {};
};
retinify::Session::Session(std::string model_path) {};
retinify::Session::~Session() {};
void retinify::Session::UploadInputData(int index, const std::vector<float> &data) {};
void retinify::Session::UploadInputData(int index, const cv::Mat &data) {};
void retinify::Session::DownloadOutputData(int index, std::vector<float> &data) {};
void retinify::Session::DownloadOutputData(int index, cv::Mat &data) {};
void retinify::Session::Run() {};
#endif
} // namespace retinify