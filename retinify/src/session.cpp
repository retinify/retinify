// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "session.hpp"
#include "mat.hpp"

#include "retinify/log.hpp"
#include "retinify/path.hpp"

#ifdef USE_NVIDIA_GPU
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <fstream>
#include <memory>
#endif

namespace retinify
{
#ifdef USE_NVIDIA_GPU
class TensorRTLogger : public nvinfer1::ILogger
{
  public:
    void log(Severity severity, const char *msg) noexcept override
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            LogFatal(msg);
            break;
        case Severity::kERROR:
            LogError(msg);
            break;
        case Severity::kWARNING:
            LogWarn(msg);
            break;
        case Severity::kINFO:
            LogInfo(msg);
            break;
        case Severity::kVERBOSE:
            LogDebug(msg);
            break;
        default:
            break;
        }
    }
};
#endif

Session::~Session() noexcept
{
#ifdef USE_NVIDIA_GPU
    delete context_;
    delete engine_;
    delete runtime_;
    if (cudaStream_ != nullptr)
    {
        (void)cudaStreamDestroy(cudaStream_);
        cudaStream_ = nullptr;
    }
#else
    if (binding_)
    {
        api_->ReleaseIoBinding(binding_);
    }
    if (runOption_)
    {
        api_->ReleaseRunOptions(runOption_);
    }
    if (session_)
    {
        api_->ReleaseSession(session_);
    }
    if (sessionOptions_)
    {
        api_->ReleaseSessionOptions(sessionOptions_);
    }
    if (deviceMemoryInfo_)
    {
        api_->ReleaseMemoryInfo(deviceMemoryInfo_);
    }
    if (env_)
    {
        api_->ReleaseEnv(env_);
    }
#endif
}

auto Session::Initialize(const char *model_path) noexcept -> Status
{
#ifdef USE_NVIDIA_GPU
    // Create CUDA stream
    cudaError_t cudaStreamError = cudaStreamCreate(&cudaStream_);
    if (cudaStreamError != cudaSuccess)
    {
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    // Create TensorRT runtime
    TensorRTLogger logger;
    runtime_ = nvinfer1::createInferRuntime(logger);
    if (runtime_ == nullptr)
    {
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    // Load TensorRT engine
    try
    {
        // Check if engine cache exists
        const char *enginePath = TensorRTEngineFilePath();
        std::ifstream engineFile(enginePath, std::ios::binary);

        if (engineFile.is_open())
        {
            engineFile.seekg(0, std::ios::end);
            const auto fileSize = engineFile.tellg();
            engineFile.seekg(0, std::ios::beg);

            if (fileSize <= 0)
            {
                return Status{StatusCategory::SYSTEM, StatusCode::FAIL};
            }

            const size_t size = static_cast<size_t>(fileSize);
            std::vector<char> engineData(size);
            if (!engineFile.read(engineData.data(), static_cast<std::streamsize>(size)))
            {
                return Status{StatusCategory::SYSTEM, StatusCode::FAIL};
            }
            engineFile.close();

            engine_ = runtime_->deserializeCudaEngine(engineData.data(), size);
        }
        else
        {
            // Build engine from ONNX model
            auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
            if (!builder)
            {
                return Status{StatusCategory::CUDA, StatusCode::FAIL};
            }

            auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
            if (!config)
            {
                return Status{StatusCategory::CUDA, StatusCode::FAIL};
            }

            auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));
            if (!network)
            {
                return Status{StatusCategory::CUDA, StatusCode::FAIL};
            }

            auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
            if (!parser)
            {
                return Status{StatusCategory::CUDA, StatusCode::FAIL};
            }

            // Parse ONNX model
            if (!parser->parseFromFile(model_path, static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
            {
                return Status{StatusCategory::CUDA, StatusCode::FAIL};
            }

            // Configure optimization
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            constexpr std::uint64_t kWorkSpacePoolSize = static_cast<std::uint64_t>(1) << 30; // 1 GiB
            config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, kWorkSpacePoolSize);

            // Set optimization profiles
            auto *profile = builder->createOptimizationProfile();
            nvinfer1::Dims minDims{4, {1, 180, 320, 3}};
            nvinfer1::Dims optDims{4, {1, 480, 640, 3}};
            nvinfer1::Dims maxDims{4, {1, 1440, 2560, 3}};

            (void)profile->setDimensions("left", nvinfer1::OptProfileSelector::kMIN, minDims);
            (void)profile->setDimensions("left", nvinfer1::OptProfileSelector::kOPT, optDims);
            (void)profile->setDimensions("left", nvinfer1::OptProfileSelector::kMAX, maxDims);
            (void)profile->setDimensions("right", nvinfer1::OptProfileSelector::kMIN, minDims);
            (void)profile->setDimensions("right", nvinfer1::OptProfileSelector::kOPT, optDims);
            (void)profile->setDimensions("right", nvinfer1::OptProfileSelector::kMAX, maxDims);

            (void)config->addOptimizationProfile(profile);

            // Build engine
            auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
            if (!serializedEngine)
            {
                return Status{StatusCategory::CUDA, StatusCode::FAIL};
            }

            // Save engine to cache
            std::ofstream engineCache(enginePath, std::ios::binary);
            if (engineCache.good())
            {
                engineCache.write(static_cast<const char *>(serializedEngine->data()), serializedEngine->size());
                engineCache.close();
            }

            // Deserialize engine
            engine_ = runtime_->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
        }
    }
    catch (...)
    {
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    if (engine_ == nullptr)
    {
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    // Create execution context
    context_ = engine_->createExecutionContext();
    if (context_ == nullptr)
    {
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    return Status{};
#else
    OrtStatus *ort_status = nullptr;

    // Create environment
    ort_status = api_->CreateEnv(ORT_LOGGING_LEVEL_FATAL, "retinify", &env_);
    if (ort_status)
    {
        api_->ReleaseStatus(ort_status);
        return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
    }

    // Create session options
    ort_status = api_->CreateSessionOptions(&sessionOptions_);
    if (ort_status)
    {
        api_->ReleaseStatus(ort_status);
        return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
    }

    // Set graph optimization level and thread count
    ort_status = api_->SetSessionGraphOptimizationLevel(sessionOptions_, ORT_ENABLE_ALL);
    if (ort_status)
    {
        api_->ReleaseStatus(ort_status);
        return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
    }

    ort_status = api_->SetIntraOpNumThreads(sessionOptions_, 4);
    if (ort_status)
    {
        api_->ReleaseStatus(ort_status);
        return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
    }

    // Create CPU memory info
    ort_status = api_->CreateMemoryInfo("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault, &deviceMemoryInfo_);
    if (ort_status)
    {
        api_->ReleaseStatus(ort_status);
        return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
    }

    // Create session
    ort_status = api_->CreateSession(env_, model_path, sessionOptions_, &session_);
    if (ort_status)
    {
        api_->ReleaseStatus(ort_status);
        return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
    }

    // Create run options
    ort_status = api_->CreateRunOptions(&runOption_);
    if (ort_status)
    {
        api_->ReleaseStatus(ort_status);
        return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
    }

    // Create IO binding
    ort_status = api_->CreateIoBinding(session_, &binding_);
    if (ort_status)
    {
        api_->ReleaseStatus(ort_status);
        return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
    }

    return Status{};
#endif
}

auto Session::BindInput(const char *name, const Mat &mat) const noexcept -> Status
{
#ifdef USE_NVIDIA_GPU
    std::array<int64_t, 4> shape = mat.Shape();
    nvinfer1::Dims dims{4, {static_cast<int>(shape[0]), static_cast<int>(shape[1]), static_cast<int>(shape[2]), static_cast<int>(shape[3])}};

    if (!context_->setInputShape(name, dims))
    {
        return Status{StatusCategory::CUDA, StatusCode::INVALID_ARGUMENT};
    }

    if (!context_->setTensorAddress(name, mat.Data()))
    {
        return Status{StatusCategory::CUDA, StatusCode::INVALID_ARGUMENT};
    }

    return Status{};
#else
    size_t elementCount = mat.ElementCount();
    size_t bytesPerElement = mat.BytesPerElement();
    std::array<int64_t, 4> shape = mat.Shape();

    OrtValue *tensor = nullptr;
    OrtStatus *ort_status = nullptr;

    ort_status = api_->CreateTensorWithDataAsOrtValue(deviceMemoryInfo_, static_cast<float *>(mat.Data()), elementCount * bytesPerElement, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &tensor);
    if (ort_status)
    {
        api_->ReleaseStatus(ort_status);
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    ort_status = api_->BindInput(binding_, name, tensor);
    if (ort_status)
    {
        api_->ReleaseStatus(ort_status);
        api_->ReleaseValue(tensor);
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    return Status{};
#endif
}

auto Session::BindOutput(const char *name, const Mat &mat) const noexcept -> Status
{
#ifdef USE_NVIDIA_GPU
    if (!context_->setTensorAddress(name, mat.Data()))
    {
        return Status{StatusCategory::CUDA, StatusCode::INVALID_ARGUMENT};
    }

    return Status{};
#else
    size_t elementCount = mat.ElementCount();
    size_t bytesPerElement = mat.BytesPerElement();
    std::array<int64_t, 4> shape = mat.Shape();

    OrtValue *tensor = nullptr;
    OrtStatus *ort_status = nullptr;

    ort_status = api_->CreateTensorWithDataAsOrtValue(deviceMemoryInfo_, static_cast<float *>(mat.Data()), elementCount * bytesPerElement, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &tensor);
    if (ort_status)
    {
        api_->ReleaseStatus(ort_status);
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    ort_status = api_->BindOutput(binding_, name, tensor);
    if (ort_status)
    {
        api_->ReleaseStatus(ort_status);
        api_->ReleaseValue(tensor);
        return Status{StatusCategory::RETINIFY, StatusCode::INVALID_ARGUMENT};
    }

    return Status{};
#endif
}

auto Session::Run() const noexcept -> Status
{
#ifdef USE_NVIDIA_GPU
    if (!context_->allInputDimensionsSpecified())
    {
        return Status{StatusCategory::CUDA, StatusCode::INVALID_ARGUMENT};
    }

    if (!context_->enqueueV3(cudaStream_))
    {
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    cudaError_t cuda_error = cudaStreamSynchronize(cudaStream_);
    if (cuda_error != cudaSuccess)
    {
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    return Status{};
#else
    OrtStatus *ort_status = nullptr;

    ort_status = api_->RunWithBinding(session_, runOption_, binding_);
    if (ort_status)
    {
        api_->ReleaseStatus(ort_status);
        return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
    }

    return Status{};
#endif
}
} // namespace retinify