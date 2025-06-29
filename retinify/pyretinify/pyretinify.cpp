// SPDX-FileCopyrightText: Copyright (c) 2025 Sensui Yagi. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "retinify/retinify.hpp"

#include <iostream>
#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

namespace pyretinify
{
static std::string PYTHON_ONNX_MODEL_PATH;

static std::string PythonONNXModelFilePath()
{
    std::filesystem::path p{PYTHON_ONNX_MODEL_PATH};
    if (!std::filesystem::exists(p))
    {
        throw std::runtime_error("ONNX model file does not exist at: " + PYTHON_ONNX_MODEL_PATH);
    }
    return PYTHON_ONNX_MODEL_PATH;
}

static std::string version()
{
    return retinify::Version();
}

PYBIND11_MODULE(pyretinify, m)
{
    py::module_ sysconfig = py::module_::import("sysconfig");
    std::string platlib = sysconfig.attr("get_path")("platlib").cast<std::string>();
    PYTHON_ONNX_MODEL_PATH = (std::filesystem::path{platlib} / "share" / "retinify" / "weights" / "model.onnx").string();

    m.def("get_onnx_model_path", &PythonONNXModelFilePath);
    m.def("version", &version);
}
} // namespace pyretinify