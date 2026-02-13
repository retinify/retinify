// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "session.hpp"
#include "mat.hpp"

#include "retinify/logging.hpp"
#include "retinify/nothrow.hpp"
#include "retinify/paths.hpp"

#include <filesystem>
#include <fstream>

#ifdef BUILD_WITH_TENSORRT
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#endif

namespace retinify
{
namespace detail
{
namespace
{
#ifdef BUILD_WITH_TENSORRT
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
} // namespace

auto Session::Initialize(const char *modelPath) noexcept -> Status
{
#ifdef BUILD_WITH_TENSORRT
    return NoThrow([&]() -> Status {
        TensorRTLogger logger;
        runtime_.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime_)
        {
            LogError("Failed to create TensorRT runtime: createInferRuntime returned nullptr");
            return Status{StatusCategory::CUDA, StatusCode::FAIL};
        }

        std::filesystem::path cacheDirectoryPath;
        auto cacheDirStatus = CacheDirectoryPath(cacheDirectoryPath);
        if (!cacheDirStatus.IsOK())
        {
            LogError("Failed to get cache directory path.");
            return cacheDirStatus;
        }

        const std::filesystem::path engineFilePath = cacheDirectoryPath / "stereo-matching.engine";

        if (std::filesystem::exists(engineFilePath))
        {
            LogInfo("Found TensorRT engine file at cache directory. Loading...");

            std::error_code ec;
            const auto fileSize = std::filesystem::file_size(engineFilePath, ec);
            if (ec || fileSize == 0)
            {
                LogError("Failed to read TensorRT engine file size.");
                return Status{StatusCategory::SYSTEM, StatusCode::FAIL};
            }

            std::ifstream file(engineFilePath, std::ios::binary);
            if (!file.is_open())
            {
                LogError("Failed to open TensorRT engine file.");
                return Status{StatusCategory::SYSTEM, StatusCode::FAIL};
            }

            std::vector<char> engineData(fileSize);
            file.read(engineData.data(), static_cast<std::streamsize>(fileSize));
            if (file.fail() || file.gcount() != static_cast<std::streamsize>(fileSize))
            {
                LogError("Failed to read TensorRT engine data from file.");
                return Status{StatusCategory::SYSTEM, StatusCode::FAIL};
            }

            engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), fileSize));
        }
        else
        {
            LogInfo("TensorRT engine not found. Starting first-time build. This process may take several minutes...");

            auto builder = std::unique_ptr<nvinfer1::IBuilder>{nvinfer1::createInferBuilder(logger)};
            if (!builder)
            {
                LogError("Failed to create TensorRT builder: createInferBuilder returned nullptr");
                return Status{StatusCategory::CUDA, StatusCode::FAIL};
            }

            auto config = std::unique_ptr<nvinfer1::IBuilderConfig>{builder->createBuilderConfig()};
            if (!config)
            {
                LogError("Failed to create TensorRT builder config: createBuilderConfig returned nullptr");
                return Status{StatusCategory::CUDA, StatusCode::FAIL};
            }

            auto network = std::unique_ptr<nvinfer1::INetworkDefinition>{builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED))};
            if (!network)
            {
                LogError("Failed to create TensorRT network definition: createNetworkV2 returned nullptr");
                return Status{StatusCategory::CUDA, StatusCode::FAIL};
            }

            auto parser = std::unique_ptr<nvonnxparser::IParser>{nvonnxparser::createParser(*network, logger)};
            if (!parser)
            {
                LogError("Failed to create ONNX parser: createParser returned nullptr");
                return Status{StatusCategory::CUDA, StatusCode::FAIL};
            }

            if (!parser->parseFromFile(modelPath, static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
            {
                LogError("Failed to parse ONNX model file.");
                return Status{StatusCategory::CUDA, StatusCode::FAIL};
            }

            constexpr std::uint64_t kWorkSpacePoolSize = static_cast<std::uint64_t>(1) << 30; // 1 GiB
            config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, kWorkSpacePoolSize);

            auto *profile = builder->createOptimizationProfile();
            if (!profile)
            {
                LogError("Failed to create optimization profile: createOptimizationProfile returned nullptr");
                return Status{StatusCategory::CUDA, StatusCode::FAIL};
            }

            nvinfer1::Dims minDims{4, {1, kEngineMinHeight, kEngineMinWidth, 1}};
            nvinfer1::Dims optDims{4, {1, kEngineOptHeight, kEngineOptWidth, 1}};
            nvinfer1::Dims maxDims{4, {1, kEngineMaxHeight, kEngineMaxWidth, 1}};

            if (!profile->setDimensions(kOnnxLeftInputName, nvinfer1::OptProfileSelector::kMIN, minDims))
            {
                LogError("Failed to set MIN dimensions for 'left' input");
                return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
            }

            if (!profile->setDimensions(kOnnxLeftInputName, nvinfer1::OptProfileSelector::kOPT, optDims))
            {
                LogError("Failed to set OPT dimensions for 'left' input");
                return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
            }

            if (!profile->setDimensions(kOnnxLeftInputName, nvinfer1::OptProfileSelector::kMAX, maxDims))
            {
                LogError("Failed to set MAX dimensions for 'left' input");
                return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
            }

            if (!profile->setDimensions(kOnnxRightInputName, nvinfer1::OptProfileSelector::kMIN, minDims))
            {
                LogError("Failed to set MIN dimensions for 'right' input");
                return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
            }

            if (!profile->setDimensions(kOnnxRightInputName, nvinfer1::OptProfileSelector::kOPT, optDims))
            {
                LogError("Failed to set OPT dimensions for 'right' input");
                return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
            }

            if (!profile->setDimensions(kOnnxRightInputName, nvinfer1::OptProfileSelector::kMAX, maxDims))
            {
                LogError("Failed to set MAX dimensions for 'right' input");
                return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
            }

            if (config->addOptimizationProfile(profile) < 0)
            {
                LogError("Failed to add optimization profile to TensorRT config");
                return Status{StatusCategory::CUDA, StatusCode::FAIL};
            }

            auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>{builder->buildSerializedNetwork(*network, *config)};
            if (!serializedEngine)
            {
                LogError("Failed to build serialized TensorRT engine: buildSerializedNetwork returned nullptr");
                return Status{StatusCategory::CUDA, StatusCode::FAIL};
            }

            std::ofstream engineFileCache(engineFilePath, std::ios::binary);
            if (engineFileCache.is_open())
            {
                engineFileCache.write(static_cast<const char *>(serializedEngine->data()), serializedEngine->size());
                engineFileCache.close();
            }

            engine_.reset(runtime_->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));
        }

        if (!engine_)
        {
            LogError("Failed to create TensorRT engine: deserializeCudaEngine returned nullptr");
            return Status{StatusCategory::CUDA, StatusCode::FAIL};
        }

        context_.reset(engine_->createExecutionContext());
        if (!context_)
        {
            LogError("Failed to create TensorRT execution context: createExecutionContext returned nullptr");
            return Status{StatusCategory::CUDA, StatusCode::FAIL};
        }

        return Status{};
    });
#else
    (void)modelPath;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto Session::BindInput(const char *name, const Mat &mat) const noexcept -> Status
{
#ifdef BUILD_WITH_TENSORRT
    std::array<int64_t, 4> shape = mat.Shape();
    nvinfer1::Dims dims{4, {static_cast<int>(shape[0]), static_cast<int>(shape[1]), static_cast<int>(shape[2]), static_cast<int>(shape[3])}};

    if (!context_->setInputShape(name, dims))
    {
        LogError("Failed to set input shape for tensor.");
        return Status{StatusCategory::CUDA, StatusCode::INVALID_ARGUMENT};
    }

    if (!context_->setTensorAddress(name, mat.Data()))
    {
        LogError("Failed to set tensor address for input.");
        return Status{StatusCategory::CUDA, StatusCode::INVALID_ARGUMENT};
    }

    return Status{};
#else
    (void)name;
    (void)mat;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto Session::BindOutput(const char *name, const Mat &mat) const noexcept -> Status
{
#ifdef BUILD_WITH_TENSORRT
    if (!context_->setTensorAddress(name, mat.Data()))
    {
        LogError("Failed to set tensor address for output.");
        return Status{StatusCategory::CUDA, StatusCode::INVALID_ARGUMENT};
    }

    return Status{};
#else
    (void)name;
    (void)mat;
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}

auto Session::Execute(Stream &stream) const noexcept -> Status
{
#ifdef BUILD_WITH_TENSORRT
    if (!context_->allInputDimensionsSpecified())
    {
        LogError("Not all input dimensions are specified.");
        return Status{StatusCategory::CUDA, StatusCode::INVALID_ARGUMENT};
    }

    if (!context_->enqueueV3(stream.GetCudaStream()))
    {
        LogError("Failed to enqueue TensorRT execution context.");
        return Status{StatusCategory::CUDA, StatusCode::FAIL};
    }

    return Status{};
#else
    LogError("This function is not available");
    return Status{StatusCategory::RETINIFY, StatusCode::FAIL};
#endif
}
} // namespace detail
} // namespace retinify
